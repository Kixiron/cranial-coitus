use crate::{
    graph::{Node, NodeId, Port, PortData, PortId, Rvsdg},
    utils::Set,
};
use std::fmt::Debug;

impl Rvsdg {
    /// Remove the given node from the graph along with all ports and edges associated with it
    #[tracing::instrument(skip(self))]
    pub fn remove_node(&mut self, node_id: NodeId) -> Option<Node> {
        if let Some(node) = self.nodes.remove(&node_id) {
            tracing::trace!("removing {:?}", node);

            // Remove all the associated ports and edges
            self.retain_graph_elements(
                |parent| parent != node_id,
                |data| data.parent != node_id,
                |parent| parent != node_id,
                |data| data.parent != node_id,
                |_, data| data.parent != node_id,
                |node| node != node_id,
            );

            Some(node)
        } else {
            tracing::trace!("could not find {:?} to remove it", node_id);

            None
        }
    }

    /// Remove all nodes mentioned in `retained_nodes` from the graph along with all
    /// of their associated ports and edges
    ///
    /// Returns `true` if any nodes, ports or edges were removed from the graph
    ///
    #[tracing::instrument(skip(self, removed_nodes))]
    pub fn bulk_remove_nodes<S>(&mut self, removed_nodes: &S) -> bool
    where
        S: Set<NodeId> + Debug,
    {
        self.retain_graph_elements(
            |parent| !removed_nodes.contains(&parent),
            |data| !removed_nodes.contains(&data.parent),
            |parent| !removed_nodes.contains(&parent),
            |data| !removed_nodes.contains(&data.parent),
            |_, data| !removed_nodes.contains(&data.parent),
            |node| !removed_nodes.contains(&node),
        )
    }

    /// Remove all nodes from the graph except for the ones mentioned in `retained_nodes`
    /// along with all of their associated ports and edges
    ///
    /// Returns `true` if any nodes, ports or edges were removed from the graph
    ///
    #[tracing::instrument(skip(self, retained_nodes))]
    pub fn bulk_retain_nodes<S>(&mut self, retained_nodes: &S) -> bool
    where
        S: Set<NodeId> + Debug,
    {
        self.retain_graph_elements(
            |parent| retained_nodes.contains(&parent),
            |data| retained_nodes.contains(&data.parent),
            |parent| retained_nodes.contains(&parent),
            |data| retained_nodes.contains(&data.parent),
            |_, data| retained_nodes.contains(&data.parent),
            |node| retained_nodes.contains(&node),
        )
    }

    /// Remove all graph elements where the given closures return `false`
    ///
    /// `forward` is run on all unique forward edge start ports, `forward_targets`
    /// is ran for each unique forward edge endpoint pair.
    /// `reverse` is run on all unique reverse edge start ports, `reverse_sources`
    /// is ran for each unique reverse edge endpoint pair.
    /// `ports` is run for each port entry.
    /// `nodes` is run for each node entry.
    ///
    /// Graph [`Start`] and [`End`] nodes are automatically updated based on the removals
    /// performed within the user functions.
    /// That is, if the `nodes` function removes any start or end nodes, the corresponding
    /// nodes will be removed from the graph's internal start and end node lists.
    ///
    /// This function will also automatically remove any edges for which any of the involved nodes
    /// no longer exists, and may also remove ports who's parent node no longer exists as well.
    ///
    pub fn retain_graph_elements<Forward, ForwardTargets, Reverse, ReverseSources, Ports, Nodes>(
        &mut self,
        mut forward: Forward,
        mut forward_targets: ForwardTargets,
        mut reverse: Reverse,
        mut reverse_sources: ReverseSources,
        mut ports: Ports,
        mut nodes: Nodes,
    ) -> bool
    where
        Forward: FnMut(NodeId) -> bool,
        ForwardTargets: FnMut(&PortData) -> bool,
        Reverse: FnMut(NodeId) -> bool,
        ReverseSources: FnMut(&PortData) -> bool,
        Ports: FnMut(PortId, &PortData) -> bool,
        Nodes: FnMut(NodeId) -> bool,
    {
        // Get the original sizes of each collection
        let (
            initial_nodes,
            initial_ports,
            initial_forward,
            initial_reverse,
            initial_start,
            initial_end,
        ) = (
            self.nodes.len(),
            self.ports.len(),
            self.forward.len(),
            self.reverse.len(),
            self.start_nodes.len(),
            self.end_nodes.len(),
        );

        // Remove forward edges
        self.forward.retain(|output, targets| {
            let parent = match self.ports.get(&output.port()) {
                Some(data) => data.parent,

                // If the port doesn't exist, remove it anyways
                None => {
                    tracing::warn!("could not get the port entry for forward edge {:?}", output);
                    return false;
                }
            };

            // If the user wants to keep this edge
            if forward(parent) {
                // Process all of the kept edge's targets
                targets.retain(|(input, _)| match self.ports.get(&input.port()) {
                    // Decide whether or not to keep the edge
                    Some(data) => forward_targets(data),

                    // If the port doesn't exist, remove it anyways
                    None => {
                        tracing::warn!(
                            ?parent,
                            "could not get the port entry for {:?}, removing forward edge from {:?} to {:?}",
                            input,
                            output,
                            input,
                        );

                        false
                    }
                });

                // If the edge has no targets, remove it
                let has_targets = !targets.is_empty();
                tracing::trace!(
                    ?parent,
                    "forward edge from {:?} has no targets, removing it",
                    output,
                );

                has_targets

            // Otherwise the user has decided to discard this edge
            } else {
                false
            }
        });

        // Remove reverse edges
        self.reverse.retain(|input, sources| {
            let parent = match self.ports.get(&input.port()) {
                Some(data) => data.parent,

                // If the port doesn't exist, remove it anyways
                None => {
                    tracing::warn!("could not get the port entry for reverse edge {:?}", input);
                    return false;
                }
            };

            // If the user wants to keep this edge
            if reverse(parent) {
                // Process all of the kept edge's sources
                sources.retain(|(output, _)| match self.ports.get(&output.port()) {
                    // Decide whether or not to keep the edge
                    Some(data) => reverse_sources(data),

                    // If the port doesn't exist, remove it anyways
                    None => {
                        tracing::warn!(
                            ?parent,
                            "could not get the port entry for {:?}, removing forward edge from {:?} to {:?}",
                            output,
                            input,
                            output,
                        );

                        false
                    }
                });

                // If the edge has no sources, remove it
                let has_sources = !sources.is_empty();
                tracing::trace!(
                    ?parent,
                    "reverse edge from {:?} has no sources, removing it",
                    input,
                );

                has_sources

            // Otherwise the user has decided to discard this edge
            } else {
                false
            }
        });

        // Remove ports
        self.ports.retain(|&port, data| {
            let should_keep = ports(port, data);
            if !should_keep {
                tracing::trace!(
                    parent = ?data.parent,
                    "removing {:?} ({} {} port)",
                    port,
                    data.edge,
                    data.kind,
                );
            }

            should_keep
        });

        // Remove nodes
        self.nodes.retain(|&node, _| {
            let should_keep = nodes(node);
            if !should_keep {
                tracing::trace!("removing {:?}", node);
            }

            should_keep
        });

        // Remove any start nodes which don't exist within the graph
        self.start_nodes.retain(|node| {
            let should_keep = self.nodes.contains_key(node);
            if !should_keep {
                tracing::trace!(
                    "removing start node {:?}, its node entry no longer exists",
                    node,
                );
            }

            should_keep
        });

        // Remove any end nodes which don't exist within the graph
        self.end_nodes.retain(|node| {
            let should_keep = self.nodes.contains_key(node);
            if !should_keep {
                tracing::trace!(
                    "removing end node {:?}, its node entry no longer exists",
                    node,
                );
            }

            should_keep
        });

        // Get the final sizes of each collection
        let (final_nodes, final_ports, final_forward, final_reverse, final_start, final_end) = (
            self.nodes.len(),
            self.ports.len(),
            self.forward.len(),
            self.reverse.len(),
            self.start_nodes.len(),
            self.end_nodes.len(),
        );

        // Return whether we've actually removed any nodes or not
        let did_modify_graph = initial_nodes != final_nodes
            || initial_ports != final_ports
            || initial_forward != final_forward
            || initial_reverse != final_reverse
            || initial_start != final_start
            || initial_end != final_end;

        tracing::trace!(
            did_modify_graph,
            final_nodes,
            final_ports,
            final_forward,
            final_reverse,
            final_start,
            final_end,
            "removed {} nodes, {} ports, {} forward edges, {} reverse edges, {} start nodes and {} end nodes",
            initial_nodes - final_nodes,
            initial_ports - final_ports,
            initial_forward - final_forward,
            initial_reverse - final_reverse,
            initial_start - final_start,
            initial_end - final_end,
        );

        did_modify_graph
    }
}
