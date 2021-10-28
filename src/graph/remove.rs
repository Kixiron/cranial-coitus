use crate::{
    graph::{Node, NodeId, Port, Rvsdg},
    utils::{Set, SingletonSet},
};
use std::fmt::Debug;

impl Rvsdg {
    /// Remove the given node from the graph along with all ports and edges associated with it
    #[tracing::instrument(skip(self))]
    pub fn remove_node(&mut self, node_id: NodeId) -> Option<Node> {
        if let Some(node) = self.nodes.remove(&node_id) {
            tracing::trace!("removing {:?}", node);

            // Remove all the associated ports and edges
            self.bulk_remove_nodes(&SingletonSet::new(node_id));

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
        if removed_nodes.is_empty() {
            tracing::trace!("got an empty set of nodes to remove, not removing any");

            return false;
        }

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
            if removed_nodes.contains(&self.ports[&output.port()].parent) {
                false
            } else {
                targets.retain(|(input, _)| {
                    self.ports.get(&input.port()).map_or_else(
                        || {
                            tracing::warn!(
                                "could not get the port entry for {:?} while bulk removing nodes, \
                                 removing forward edge from {:?} to {:?}",
                                input,
                                output,
                                input,
                            );

                            false
                        },
                        |data| {
                            let should_keep = removed_nodes.contains(&data.parent);
                            if !should_keep {
                                tracing::trace!(
                                    "removing forward edge from {:?} to {:?} ({:?}'s parent is {:?}) \
                                     while bulk removing nodes",
                                    output,
                                    output,
                                    input,
                                    data.parent,
                                );
                            }
                            debug_assert!(data.kind.is_input(), "{:?} is not an input port", output);

                            should_keep
                        },
                    )
                });

                !targets.is_empty()
            }
        });

        // Remove reverse edges
        self.reverse.retain(|input, sources| {
            if removed_nodes.contains(&self.ports[&input.port()].parent) {
                false
            } else {
                sources.retain(|(output, _, )| {
                    self.ports.get(&output.port()).map_or_else(
                        || {
                            tracing::warn!(
                                "could not get the port entry for {:?} while bulk removing nodes, \
                                 removing reverse edge from {:?} to {:?}",
                                output,
                                input,
                                output,
                            );

                            false
                        },
                        |data| {
                            let should_keep = !removed_nodes.contains(&data.parent);
                            if !should_keep {
                                tracing::trace!(
                                    "removing forward edge from {:?} to {:?} ({:?}'s parent is {:?}) \
                                     while bulk removing nodes",
                                    output,
                                    output,
                                    input,
                                    data.parent,
                                );
                            }
                            debug_assert!(data.kind.is_output(), "{:?} is not an output port", output);

                            should_keep
                        },
                    )
                });

                !sources.is_empty()
            }
        });

        // Remove ports
        self.ports.retain(|port, data| {
            let should_keep = !removed_nodes.contains(&data.parent);
            if !should_keep {
                tracing::trace!(
                    parent = ?data.parent,
                    "removing {:?} ({} {} port) while bulk removing nodes",
                    port,
                    data.edge,
                    data.kind,
                );
            }

            should_keep
        });

        // Remove nodes
        self.nodes.retain(|node, _| {
            let should_keep = !removed_nodes.contains(node);
            if !should_keep {
                tracing::trace!("removing {:?} while bulk removing nodes", node);
            }

            should_keep
        });

        self.start_nodes.retain(|node| {
            let should_keep = self.nodes.contains_key(node);
            if !should_keep {
                tracing::trace!("removing start node {:?} while bulk removing nodes", node);
            }

            should_keep
        });

        self.end_nodes.retain(|node| {
            let should_keep = self.nodes.contains_key(node);
            if !should_keep {
                tracing::trace!("removing end node {:?} while bulk removing nodes", node);
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

        tracing::trace!(
            final_nodes, final_ports, final_forward, final_reverse, final_start, final_end,
            "removed {} nodes, {} ports, {} forward edges, {} reverse edges, {} start nodes and {} end nodes",
            initial_nodes - final_nodes,
            initial_ports - final_ports,
            initial_forward - final_forward,
            initial_reverse - final_reverse,
            initial_start - final_start,
            initial_end - final_end,
        );

        // Return whether we've actually removed any nodes or not
        initial_nodes != final_nodes
            || initial_ports != final_ports
            || initial_forward != final_forward
            || initial_reverse != final_reverse
            || initial_start != final_start
            || initial_end != final_end
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

        // If we've been instructed to retain zero nodes, just clear the graph
        if retained_nodes.is_empty() {
            tracing::trace!(
                removed_forward_edges = initial_nodes,
                removed_reverse_edges = initial_ports,
                removed_ports = initial_forward,
                removed_nodes = initial_reverse,
                removed_start_nodes = initial_start,
                removed_end_nodes = initial_end,
                "got an empty set of nodes to retain, removing all nodes, ports and edges from the graph",
            );

            // If there's actually any elements within the graph
            let graph_is_empty = initial_nodes != 0
                || initial_ports != 0
                || initial_forward != 0
                || initial_reverse != 0
                || initial_start != 0
                || initial_end != 0;

            if !graph_is_empty {
                self.forward.clear();
                self.reverse.clear();
                self.ports.clear();
                self.nodes.clear();
                self.start_nodes.clear();
                self.end_nodes.clear();
            }

            return graph_is_empty;
        }

        // Remove forward edges
        self.forward.retain(|output, targets| {
            if retained_nodes.contains(&self.ports[&output.port()].parent) {
                targets.retain(|(input, _)| {
                    self.ports.get(&input.port()).map_or_else(
                        || {
                            tracing::warn!(
                                "could not get the port entry for {:?} while bulk retaining nodes, \
                                 removing forward edge from {:?} to {:?}",
                                input,
                                output,
                                input,
                            );

                            false
                        },
                        |data| {
                            let should_keep = retained_nodes.contains(&data.parent);
                            if !should_keep {
                                tracing::trace!(
                                    "removing forward edge from {:?} to {:?} ({:?}'s parent is {:?}) \
                                     while bulk retaining nodes",
                                    output,
                                    output,
                                    input,
                                    data.parent,
                                );
                            }
                            debug_assert!(data.kind.is_input(), "{:?} is not an input port", output);

                            should_keep
                        },
                    )
                });

                if targets.is_empty() {
                    tracing::trace!(
                        "found that forward edge entry for {:?} was empty while bulk retaining nodes",
                        output,
                    );
                }

                !targets.is_empty()
            } else {
                false
            }
        });

        // Remove reverse edges
        self.reverse.retain(|input, sources| {
            if retained_nodes.contains(&self.ports[&input.port()].parent) {
                sources.retain(|(output, _, )| {
                    self.ports.get(&output.port()).map_or_else(
                        || {
                            tracing::warn!(
                                "could not get the port entry for {:?} while bulk retaining nodes, \
                                 removing reverse edge from {:?} to {:?}",
                                output,
                                input,
                                output,
                            );

                            false
                        },
                        |data| {
                            let should_keep = retained_nodes.contains(&data.parent);
                            if !should_keep {
                                tracing::trace!(
                                    "removing reverse edge from {:?} to {:?} ({:?}'s parent is {:?}) \
                                     while bulk retaining nodes",
                                    input,
                                    output,
                                    output,
                                    data.parent,
                                );
                            }
                            debug_assert!(data.kind.is_output(), "{:?} is not an output port", output);

                            should_keep
                        },
                    )
                });

                if sources.is_empty() {
                    tracing::trace!(
                        "found that reverse edge entry for {:?} was empty while bulk retaining nodes",
                        input,
                    );
                }

                !sources.is_empty()
            } else {
                false
            }
        });

        // Remove ports
        self.ports.retain(|port, data| {
            let should_keep = retained_nodes.contains(&data.parent);
            if !should_keep {
                tracing::trace!(
                    parent = ?data.parent,
                    "removing {:?} ({} {} port) while bulk retaining nodes",
                    port,
                    data.edge,
                    data.kind,
                );
            }

            should_keep
        });

        // Remove nodes
        self.nodes.retain(|node, _| {
            let should_keep = retained_nodes.contains(node);
            if !should_keep {
                tracing::trace!("removing {:?} while bulk retaining nodes", node);
            }

            should_keep
        });

        self.start_nodes.retain(|node| {
            let should_keep = self.nodes.contains_key(node);
            if !should_keep {
                tracing::trace!("removing start node {:?} while bulk retaining nodes", node);
            }

            should_keep
        });

        self.end_nodes.retain(|node| {
            let should_keep = self.nodes.contains_key(node);
            if !should_keep {
                tracing::trace!("removing end node {:?} while bulk retaining nodes", node);
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

        tracing::trace!(
            final_nodes, final_ports, final_forward, final_reverse, final_start, final_end,
            "removed {} nodes, {} ports, {} forward edges, {} reverse edges, {} start nodes and {} end nodes",
            initial_nodes - final_nodes,
            initial_ports - final_ports,
            initial_forward - final_forward,
            initial_reverse - final_reverse,
            initial_start - final_start,
            initial_end - final_end,
        );

        // Return whether we've actually removed any nodes or not
        initial_nodes != final_nodes
            || initial_ports != final_ports
            || initial_forward != final_forward
            || initial_reverse != final_reverse
            || initial_start != final_start
            || initial_end != final_end
    }
}
