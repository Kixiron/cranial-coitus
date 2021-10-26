mod gamma;
mod node;
mod node_ext;
mod ports;
mod stats;
mod subgraph;
mod theta;

pub use gamma::{Gamma, GammaData};
pub use node::Node;
pub use node_ext::{EdgeCount, EdgeDescriptor, NodeExt};
pub use ports::{InputPort, OutputPort, Port, PortId};
pub use subgraph::Subgraph;
pub use theta::{Theta, ThetaData};

use crate::{
    graph::{
        gamma::GammaStub,
        theta::{ThetaEffects, ThetaStub},
    },
    utils::AssertNone,
};
use std::{
    cell::Cell,
    collections::{btree_map::Entry, BTreeMap, BTreeSet, HashSet},
    fmt::{self, Debug, Display, Write},
    hash::Hash,
    rc::Rc,
};
use tinyvec::TinyVec;

// FIXME: Make a `Subgraph` type and use it in gamma & theta nodes
// TODO: Automatically track graph changes within the rvsdg itself
//       to remove the need for manual `.changed()` calls
// TODO: Make a manual iterator for nodes
// TODO: Sorting & binary search for `EdgeData` vectors?
// TODO: Use custom hasher for HashMaps

type Counter<T> = Rc<Cell<T>>;
// TODO: TinySet?
type EdgeData<T> = TinyVec<[(T, EdgeKind); 2]>;

// TODO: Structural equivalence method
#[derive(Debug, Clone, PartialEq)]
pub struct Rvsdg {
    nodes: BTreeMap<NodeId, Node>,
    forward: BTreeMap<OutputPort, EdgeData<InputPort>>,
    reverse: BTreeMap<InputPort, EdgeData<OutputPort>>,
    ports: BTreeMap<PortId, PortData>,
    node_counter: Counter<NodeId>,
    port_counter: Counter<PortId>,
}

impl Rvsdg {
    pub fn new() -> Self {
        Self::from_counters(
            Rc::new(Cell::new(NodeId::new(0))),
            Rc::new(Cell::new(PortId::new(0))),
        )
    }

    fn from_counters(node_counter: Counter<NodeId>, port_counter: Counter<PortId>) -> Self {
        Self {
            nodes: BTreeMap::new(),
            forward: BTreeMap::new(),
            reverse: BTreeMap::new(),
            ports: BTreeMap::new(),
            node_counter,
            port_counter,
        }
    }

    fn next_node(&mut self) -> NodeId {
        let node = self.node_counter.get();
        self.node_counter.set(NodeId::new(node.0 + 1));

        node
    }

    fn next_port(&mut self) -> PortId {
        let port = self.port_counter.get();
        self.port_counter.set(PortId::new(port.raw() + 1));

        port
    }

    #[track_caller]
    pub fn add_edge(&mut self, src: OutputPort, dest: InputPort, kind: EdgeKind) {
        tracing::trace!(
            source_port = ?self.ports.get(&src.port()),
            dest_port = ?self.ports.get(&dest.port()),
            edge_kind = ?kind,
            "adding {} edge from {:?} to {:?}",
            kind,
            src,
            dest,
        );

        if cfg!(debug_assertions) {
            assert!(self.ports.contains_key(&src.port()));
            assert!(self.ports.contains_key(&dest.port()));

            let src_data = self.ports[&src.port()];
            let dest_data = self.ports[&dest.port()];

            assert_eq!(
                src_data.kind,
                PortKind::Output,
                "expected source port {:?} to be an output port",
                src.port(),
            );
            assert_eq!(
                dest_data.kind,
                PortKind::Input,
                "expected destination port {:?} to be an input port",
                src.port(),
            );
            assert_eq!(
                src_data.edge, kind,
                "source port {:?} has wrong edge kind (adding {} edge \
                 from {:?} to {:?}, between nodes {:?} and {:?})",
                src, kind, src, dest, src_data.parent, dest_data.parent,
            );
            assert_eq!(
                dest_data.edge, kind,
                "destination port {:?} has wrong edge kind (adding {} edge \
                 from {:?} to {:?}, between nodes {:?} and {:?})",
                dest, kind, src, dest, src_data.parent, dest_data.parent,
            );

            assert_ne!(
                src_data.parent, dest_data.parent,
                "cannot create an edge from a node to itself",
            );
        }

        tracing::trace!("added forward edge from {:?} to {:?}", src, dest);
        self.forward
            .entry(src)
            .and_modify(|data| {
                debug_assert!(
                    !data.contains(&(dest, kind)),
                    "edges cannot point to themselves",
                );
                data.push((dest, kind));
            })
            .or_insert_with(|| {
                let mut data = TinyVec::with_capacity(1);
                data.push((dest, kind));
                data
            });

        tracing::trace!("added reverse edge from {:?} to {:?}", dest, src,);
        self.reverse
            .entry(dest)
            .and_modify(|data| {
                debug_assert!(
                    !data.contains(&(src, kind)),
                    "edges cannot point to themselves",
                );
                data.push((src, kind));
            })
            .or_insert_with(|| {
                let mut data = TinyVec::with_capacity(1);
                data.push((src, kind));
                data
            });
    }

    #[track_caller]
    pub fn add_value_edge(&mut self, src: OutputPort, dest: InputPort) {
        self.add_edge(src, dest, EdgeKind::Value);
    }

    #[track_caller]
    pub fn add_effect_edge(&mut self, src: OutputPort, dest: InputPort) {
        self.add_edge(src, dest, EdgeKind::Effect);
    }

    #[allow(dead_code)]
    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn contains_node(&self, node: NodeId) -> bool {
        self.nodes.contains_key(&node)
    }

    pub fn add_node(&mut self, node_id: NodeId, node: Node) {
        self.nodes.insert(node_id, node).debug_unwrap_none();
    }

    pub fn input_port(&mut self, parent: NodeId, edge: EdgeKind) -> InputPort {
        let port = self.next_port();
        self.ports
            .insert(port, PortData::input(parent, edge))
            .debug_unwrap_none();

        InputPort::new(port)
    }

    pub fn output_port(&mut self, parent: NodeId, edge: EdgeKind) -> OutputPort {
        let port = self.next_port();
        self.ports
            .insert(port, PortData::output(parent, edge))
            .debug_unwrap_none();

        OutputPort::new(port)
    }

    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.keys().copied()
    }

    pub fn port_ids(&self) -> impl Iterator<Item = PortId> + '_ {
        self.ports.keys().copied()
    }

    #[track_caller]
    pub fn iter_nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> + '_ {
        self.nodes
            .keys()
            .copied()
            .map(|node_id| (node_id, self.get_node(node_id)))
    }

    /// Collect all nodes from the current graph and all subgraphs
    #[track_caller]
    #[allow(dead_code)]
    pub fn transitive_nodes(&self) -> Vec<&Node> {
        let (mut buffer, mut visited) = (Vec::new(), BTreeSet::new());
        self.transitive_nodes_into(&mut buffer, &mut visited);

        buffer
    }

    /// Collect all nodes from the current graph and all subgraphs into a buffer
    #[track_caller]
    pub fn transitive_nodes_into<'a>(
        &'a self,
        buffer: &mut Vec<&'a Node>,
        visited: &mut BTreeSet<NodeId>,
    ) {
        buffer.reserve(self.nodes.len());

        for node in self.nodes.values() {
            // If this node hasn't been added already
            if visited.insert(node.node_id()) {
                // Add the current node to the buffer
                buffer.push(node);

                // Add the nodes from any subgraphs to the buffer
                match node {
                    Node::Gamma(gamma) => {
                        gamma.true_branch().transitive_nodes_into(buffer, visited);
                        gamma.false_branch().transitive_nodes_into(buffer, visited);
                    }
                    Node::Theta(theta) => theta.body().transitive_nodes_into(buffer, visited),

                    _ => {}
                }
            }
        }
    }

    pub fn try_node(&self, node: NodeId) -> Option<&Node> {
        self.nodes.get(&node)
    }

    #[track_caller]
    pub fn get_node(&self, node: NodeId) -> &Node {
        if let Some(node) = self.try_node(node) {
            node
        } else {
            panic!("tried to get node that doesn't exist: {:?}", node)
        }
    }

    #[allow(dead_code)]
    pub fn try_node_mut(&mut self, node: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&node)
    }

    #[track_caller]
    #[allow(dead_code)]
    pub fn get_node_mut(&mut self, node: NodeId) -> &mut Node {
        if let Some(node) = self.try_node_mut(node) {
            node
        } else {
            panic!("tried to get node that doesn't exist: {:?}", node)
        }
    }

    #[track_caller]
    pub fn port_parent<P: Port>(&self, port: P) -> NodeId {
        self.ports[&port.port()].parent
    }

    pub fn incoming_count(&self, node: NodeId) -> usize {
        self.get_node(node)
            .inputs()
            .into_iter()
            .flat_map(|input| self.inputs_inner(input).map(|(.., count)| count as usize))
            .sum()
    }

    pub fn value_input_count(&self, node: NodeId) -> usize {
        self.get_node(node)
            .inputs()
            .into_iter()
            .map(|input| {
                self.inputs_inner(input)
                    .filter(|(.., kind)| kind.is_value())
                    .count()
            })
            .sum()
    }

    pub fn effect_input_count(&self, node: NodeId) -> usize {
        self.get_node(node)
            .inputs()
            .into_iter()
            .map(|input| {
                self.inputs_inner(input)
                    .filter(|(.., kind)| kind.is_effect())
                    .count()
            })
            .sum()
    }

    pub fn outgoing_count(&self, node: NodeId) -> usize {
        self.get_node(node)
            .outputs()
            .into_iter()
            .flat_map(|output| self.outputs_inner(output).map(|(.., count)| count as usize))
            .sum()
    }

    pub fn value_output_count(&self, node: NodeId) -> usize {
        self.get_node(node)
            .outputs()
            .into_iter()
            .map(|output| {
                self.outputs_inner(output)
                    .filter(|(.., kind)| kind.is_value())
                    .count()
            })
            .sum()
    }

    pub fn effect_output_count(&self, node: NodeId) -> usize {
        self.get_node(node)
            .outputs()
            .into_iter()
            .map(|output| {
                self.outputs_inner(output)
                    .filter(|(.., kind)| kind.is_effect())
                    .count()
            })
            .sum()
    }

    #[inline]
    #[track_caller]
    fn outputs_inner(
        &self,
        output: OutputPort,
    ) -> impl Iterator<Item = (&Node, InputPort, OutputPort, EdgeKind)> + Clone + '_ {
        if cfg!(debug_assertions) {
            if let Some(port_data) = self.ports.get(&output.port()) {
                assert_eq!(port_data.kind, PortKind::Output);
                assert!(self.nodes.contains_key(&port_data.parent));
            }
        }

        self.forward
            .get(&output)
            .into_iter()
            .flat_map(move |incoming| {
                incoming.iter().filter_map(move |&(consumer, edge_kind)| {
                    if cfg!(debug_assertions) && self.ports.contains_key(&consumer.port()) {
                        let port_data = self.ports[&consumer.port()];
                        assert_eq!(port_data.kind, PortKind::Input);
                        assert!(self.nodes.contains_key(&port_data.parent));
                    }

                    let port_data = self.ports.get(&consumer.port())?;
                    let consumer_node = self.nodes.get(&port_data.parent)?;

                    Some((consumer_node, consumer, output, edge_kind))
                })
            })
    }

    pub fn output_dest(&self, output: OutputPort) -> impl Iterator<Item = InputPort> + Clone + '_ {
        self.ports
            .get(&output.port())
            .copied()
            .and_then(|data| {
                debug_assert_eq!(data.kind, PortKind::Output);
                debug_assert!(self.nodes.contains_key(&data.parent));

                self.forward.get(&output)
            })
            .into_iter()
            .flat_map(move |outgoing| {
                outgoing.iter().map(move |&(target, ..)| {
                    debug_assert_eq!(self.ports[&target.port()].kind, PortKind::Input);

                    target
                })
            })
    }

    #[inline]
    #[track_caller]
    fn inputs_inner(
        &self,
        input: InputPort,
    ) -> impl Iterator<Item = (&Node, InputPort, OutputPort, EdgeKind)> + Clone + '_ {
        if cfg!(debug_assertions) {
            if let Some(port_data) = self.ports.get(&input.port()) {
                assert_eq!(port_data.kind, PortKind::Input);
                assert!(self.nodes.contains_key(&port_data.parent));
            }
        }

        self.reverse
            .get(&input)
            .into_iter()
            .flat_map(move |incoming| {
                incoming.iter().filter_map(move |&(source, edge_kind)| {
                    if cfg!(debug_assertions) && self.ports.contains_key(&source.port()) {
                        let port_data = self.ports[&source.port()];
                        assert_eq!(port_data.kind, PortKind::Output);
                        assert!(self.nodes.contains_key(&port_data.parent));
                    }

                    let port_data = self.ports.get(&source.port())?;
                    let source_node = self.nodes.get(&port_data.parent)?;

                    Some((source_node, input, source, edge_kind))
                })
            })
    }

    #[track_caller]
    pub fn get_input(&self, input: InputPort) -> (&Node, OutputPort, EdgeKind) {
        match self.try_input(input) {
            Some(input) => input,
            None => {
                let port = self.ports[&input.port()];

                panic!(
                    "missing input edge for {:?} (port: {:?}, parent: {:?})",
                    input, port, self.nodes[&port.parent],
                );
            }
        }
    }

    #[track_caller]
    pub fn input_source_node(&self, input: InputPort) -> &Node {
        self.get_input(input).0
    }

    #[track_caller]
    pub fn try_input(&self, input: InputPort) -> Option<(&Node, OutputPort, EdgeKind)> {
        let (parent_node, _, source_port, edge_kind) = self.inputs_inner(input).next()?;

        Some((parent_node, source_port, edge_kind))
    }

    // TODO: Invariant assertions
    #[track_caller]
    pub fn try_input_source(&self, input: InputPort) -> Option<OutputPort> {
        self.reverse
            .get(&input)
            .and_then(|sources| sources.get(0))
            .map(|&(source, ..)| source)
    }

    // TODO: Invariant assertions
    #[track_caller]
    pub fn input_source(&self, input: InputPort) -> OutputPort {
        if let Some(source) = self.try_input_source(input) {
            source
        } else {
            let guess = if cfg!(debug_assertions) && !self.ports.contains_key(&input.port()) {
                " (guess: its port entry doesn't exist)"
            } else {
                ""
            };

            panic!("failed to get input source for {:?}{}", input, guess);
        }
    }

    #[track_caller]
    pub fn to_node<T>(&self, node: NodeId) -> T
    where
        for<'a> &'a Node: TryInto<T>,
    {
        if let Some(node) = self.cast_node::<T>(node) {
            node
        } else {
            panic!("failed to cast node to {}", std::any::type_name::<T>())
        }
    }

    pub fn cast_node<T>(&self, node: NodeId) -> Option<T>
    where
        for<'a> &'a Node: TryInto<T>,
    {
        self.try_node(node).and_then(|node| node.try_into().ok())
    }

    pub fn cast_source<T>(&self, target: InputPort) -> Option<T>
    where
        for<'a> &'a Node: TryInto<T>,
    {
        self.try_input(target)
            .and_then(|(node, _, _)| node.try_into().ok())
    }

    pub fn cast_target<T>(&self, source: OutputPort) -> Option<T>
    where
        for<'a> &'a Node: TryInto<T>,
    {
        self.get_output(source)
            .and_then(|(node, _, _)| node.try_into().ok())
    }

    pub fn get_output(&self, output: OutputPort) -> Option<(&Node, InputPort, EdgeKind)> {
        self.get_outputs(output).next()
    }

    pub fn get_outputs(
        &self,
        output: OutputPort,
    ) -> impl Iterator<Item = (&Node, InputPort, EdgeKind)> {
        if cfg!(debug_assertions) {
            if let Some(port_data) = self.ports.get(&output.port()) {
                assert_eq!(port_data.kind, PortKind::Output);
                assert!(self.nodes.contains_key(&port_data.parent));
            }
        }

        self.forward
            .get(&output)
            .into_iter()
            .flat_map(move |incoming| {
                incoming.iter().filter_map(move |&(input, edge_kind)| {
                    if cfg!(debug_assertions) && self.ports.contains_key(&input.port()) {
                        let port_data = self.ports[&input.port()];
                        assert_eq!(port_data.kind, PortKind::Input);
                        assert!(self.nodes.contains_key(&port_data.parent));
                    }

                    let port_data = self.ports.get(&input.port())?;
                    let source_node = self.nodes.get(&port_data.parent)?;

                    Some((source_node, input, edge_kind))
                })
            })
    }

    #[track_caller]
    pub fn inputs(
        &self,
        node: NodeId,
    ) -> impl Iterator<Item = (InputPort, &Node, OutputPort, EdgeKind)> {
        let node = self.get_node(node);

        // TODO: Remove need for `.inputs()` call & allocation here
        node.inputs().into_iter().flat_map(|input| {
            self.inputs_inner(input)
                .map(|(node, input, output, edge)| (input, node, output, edge))
        })
    }

    #[allow(dead_code)]
    pub fn try_inputs(
        &self,
        node: NodeId,
    ) -> impl Iterator<Item = (InputPort, Option<(&Node, OutputPort, EdgeKind)>)> {
        let node = self.get_node(node);

        node.inputs()
            .into_iter()
            .map(move |input| (input, self.try_input(input)))
    }

    #[track_caller]
    #[deprecated = "this is terrible and lies to the user"]
    pub fn outputs(
        &self,
        node: NodeId,
    ) -> impl Iterator<Item = (OutputPort, Option<(&Node, InputPort, EdgeKind)>)> {
        self.get_node(node)
            .outputs()
            .into_iter()
            .map(|output| (output, self.get_output(output)))
    }

    // FIXME: Invariant assertions and logging
    #[tracing::instrument(skip(self))]
    pub fn remove_input_edges(&mut self, input: InputPort) {
        debug_assert_eq!(self.ports[&input.port()].kind, PortKind::Input);

        if let Entry::Occupied(edges) = self.reverse.entry(input) {
            let edges = edges.remove();
            tracing::trace!("removing {} reverse edges from {:?}", edges.len(), input);

            for (output, kind) in edges {
                tracing::trace!(
                    "removing reverse {} edge from {:?} to {:?}",
                    kind,
                    input,
                    output,
                );

                if let Entry::Occupied(mut forward) = self.forward.entry(output) {
                    let mut idx = 0;
                    while idx < forward.get().len() {
                        if forward.get()[idx].0 == input {
                            let (input, kind) = forward.get_mut().remove(idx);

                            tracing::trace!(
                                "removing forward {} edge from {:?} to {:?}",
                                kind,
                                output,
                                input,
                            );
                        } else {
                            idx += 1;
                        }
                    }
                }
            }
        } else {
            tracing::trace!("removing 0 reverse edges from {:?}", input);
        }
    }

    // FIXME: Invariant assertions and logging
    #[tracing::instrument(skip(self))]
    pub fn remove_output_edges(&mut self, output: OutputPort) {
        debug_assert_eq!(self.ports[&output.port()].kind, PortKind::Output);

        if let Entry::Occupied(edges) = self.forward.entry(output) {
            let edges = edges.remove();
            tracing::trace!("removing {} forward edges from {:?}", edges.len(), output);

            for (input, kind) in edges {
                tracing::trace!(
                    "removing forward {} edge from {:?} to {:?}",
                    kind,
                    output,
                    input,
                );

                if let Entry::Occupied(mut reverse) = self.reverse.entry(input) {
                    let mut idx = 0;
                    while idx < reverse.get().len() {
                        if reverse.get()[idx].0 == output {
                            let (output, kind) = reverse.get_mut().remove(idx);

                            tracing::trace!(
                                "removing reverse {} edge from {:?} to {:?}",
                                kind,
                                input,
                                output,
                            );
                        } else {
                            idx += 1;
                        }
                    }
                } else {
                    tracing::trace!(
                        "removing 0 forward edges from {:?} \
                         (reason: could not get forward edges for {:?})",
                        output,
                        input,
                    );
                }
            }
        } else {
            tracing::trace!(
                "removing 0 forward edges from {:?} \
                 (reason: could not get forward edges for {:?})",
                output,
                output,
            );
        }
    }

    #[tracing::instrument(skip(self))]
    pub fn remove_output_port(&mut self, output: OutputPort) {
        tracing::trace!(
            "removing output port {:?} from {:?}",
            output,
            self.port_parent(output),
        );

        self.remove_output_edges(output);
        self.ports.remove(&output.port());
    }

    #[tracing::instrument(skip(self))]
    pub fn remove_inputs(&mut self, node: NodeId) {
        self.forward.retain(|output, consumers| {
            consumers.retain(|(input, kind)| {
                self.ports.get(&input.port()).map_or_else(
                    || {
                        tracing::warn!(
                            "could not get the port entry for {:?} while \
                             removing inputs for {:?}, removing",
                            input,
                            node,
                        );

                        false
                    },
                    |data| {
                        if data.parent != node {
                            true
                        } else {
                            tracing::trace!(
                                "removing forward {} edge from {:?} to {:?} while \
                                 removing inputs for {:?}",
                                kind,
                                output,
                                input,
                                node,
                            );

                            false
                        }
                    },
                )
            });

            if consumers.is_empty() {
                tracing::trace!(
                    "removing forward edge entry for {:?}, it has no consumers \
                     (happened while removing inputs for {:?})",
                    output,
                    node,
                );

                false
            } else {
                true
            }
        });

        self.reverse
            .retain(|input, _| self.ports[&input.port()].parent != node);
    }

    #[tracing::instrument(skip(self))]
    pub fn remove_outputs(&mut self, node: NodeId) {
        self.forward
            .retain(|output, _| self.ports[&output.port()].parent != node);

        self.reverse.retain(|input, sources| {
            sources.retain(|(output, kind, )| {
                self.ports
                    .get(&output.port())
                    .map_or_else(
                        || {
                            tracing::warn!(
                                "could not get the port entry for {:?} while removing outputs for {:?}, \
                                 removing reverse edge from {:?} to {:?}",
                                output,
                                node,
                                output,
                                input,
                            );

                            false
                        },
                        |data| if data.parent != node {
                            true
                        } else {
                            tracing::trace!(
                                "removing reverse {} edge from {:?} to {:?} while removing outputs for {:?}",
                                kind,
                                input,
                                output,
                                node,
                            );

                            false
                        },
                    )
            });

            if sources.is_empty() {
                tracing::trace!(
                    "removing reverse edge entry for {:?}, it has no sources \
                     (happened while removing outputs for {:?})",
                    input,
                    node,
                );

                false
            } else {
                true
            }
        });
    }

    pub fn replace_node<N>(&mut self, node_id: NodeId, node: N)
    where
        N: Into<Node>,
    {
        tracing::trace!("replacing node {:?}", node_id);

        *self
            .nodes
            .get_mut(&node_id)
            .expect("attempted to replace a node that doesn't exist") = node.into();
    }

    #[tracing::instrument(skip(self))]
    pub fn remove_node(&mut self, node: NodeId) {
        /*
        tracing::trace!("removing node {:?}", node);

        let removed_ports = self.ports.drain_filter(|_, data| data.parent == node);
        for (port, data) in removed_ports {
            tracing::trace!(
                "removed child port of {:?} {}({})",
                node,
                match data.kind {
                    PortKind::Input => "InputPort",
                    PortKind::Output => "OutputPort",
                },
                port,
            );

            match data.kind {
                PortKind::Input => {
                    if let Some(removed) = self.reverse.remove(&InputPort::new(port)) {
                        tracing::trace!(
                            reverse_edges = ?removed,
                            "removing {} reverse edges from InputPort({})",
                            removed.len(),
                            port,
                        );
                    } else {
                        tracing::trace!("InputPort({}) had no reverse edges", port);
                    }
                }
                PortKind::Output => {
                    if let Some(removed) = self.forward.remove(&OutputPort::new(port)) {
                        tracing::trace!(
                            reverse_edges = ?removed,
                            "removing {} forward edges from OutputPort({})",
                            removed.len(),
                            port,
                        );
                    } else {
                        tracing::trace!("OutputPort({}) had no forward edges", port);
                    }
                }
            }
        }

        self.nodes.remove(&node);
        */

        // TODO: Remove allocation
        let mut set = HashSet::with_capacity(1);
        set.insert(node);
        self.bulk_remove_nodes(&set);
    }

    #[tracing::instrument(skip(self))]
    pub fn bulk_remove_nodes(&mut self, nodes: &HashSet<NodeId>) {
        // Remove forward edges
        self.forward.retain(|output, targets| {
            if nodes.contains(&self.ports[&output.port()].parent) {
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
                        |data| !nodes.contains(&data.parent),
                    )
                });

                !targets.is_empty()
            }
        });

        // Remove reverse edges
        self.reverse.retain(|input, sources| {
            if nodes.contains(&self.ports[&input.port()].parent) {
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
                            debug_assert!(data.kind.is_output(), "{:?} is not an output port", output);

                            if nodes.contains(&data.parent) {
                                tracing::trace!(
                                    "removing reverse edge from {:?} to {:?} ({:?}'s parent is {:?})",
                                    input,
                                    output,
                                    output,
                                    data.parent,
                                );

                                false
                            } else {
                                true
                            }
                        },
                    )
                });

                !sources.is_empty()
            }
        });

        // Remove ports
        self.ports.retain(|port, data| {
            if nodes.contains(&data.parent) {
                tracing::trace!(
                    parent = ?data.parent,
                    "removing {:?} ({} {} port) while bulk removing nodes",
                    port,
                    data.edge,
                    data.kind,
                );

                false
            } else {
                true
            }
        });

        // Remove nodes
        self.nodes.retain(|node, _| {
            if nodes.contains(node) {
                tracing::trace!("removing {:?} while bulk removing nodes", node);

                false
            } else {
                true
            }
        });
    }

    // FIXME: Invariant assertions & logging
    #[track_caller]
    #[tracing::instrument(skip(self))]
    pub fn rewire_dependents(&mut self, old_port: OutputPort, rewire_to: OutputPort) {
        tracing::trace!(
            old = ?self.ports.get(&old_port.port()),
            new = ?self.ports.get(&rewire_to.port()),
            "rewiring dependents from {:?} to {:?}",
            old_port,
            rewire_to,
        );

        debug_assert!(self.ports.contains_key(&old_port.port()));
        debug_assert!(self.ports.contains_key(&rewire_to.port()));
        debug_assert_eq!(self.ports[&old_port.port()].kind, PortKind::Output);
        debug_assert_eq!(self.ports[&rewire_to.port()].kind, PortKind::Output);
        debug_assert_eq!(
            self.ports[&old_port.port()].edge,
            self.ports[&rewire_to.port()].edge,
        );

        let removed = if let Entry::Occupied(forward) = self.forward.entry(old_port) {
            let edges = forward.remove();
            tracing::trace!(
                forward_edges = ?edges,
                "found {} forward edges going to {:?}",
                edges.len(),
                old_port,
            );

            Some(edges)
        } else {
            tracing::trace!(
                "{:?} has no forward edges so we can't rewire anything to {:?}",
                old_port,
                rewire_to,
            );

            None
        };

        // TODO: Sorting could help with speed here?
        if let Some(forward_edges) = removed {
            for &(input, kind) in &forward_edges {
                tracing::trace!(
                    "rewiring {} edge from {:?}->{:?} into {:?}->{:?}",
                    kind,
                    old_port,
                    input,
                    rewire_to,
                    input,
                );
                debug_assert_eq!(self.ports[&old_port.port()].edge, kind);

                if let Entry::Occupied(mut reverse) = self.reverse.entry(input) {
                    tracing::trace!(
                        reverse_edges = ?reverse.get(),
                        "got {} reverse edges for {:?} while rewiring from {:?} to {:?}",
                        reverse.get().len(),
                        input,
                        old_port,
                        rewire_to,
                    );

                    for (output, edge_kind) in reverse.get_mut() {
                        if *output == old_port && *edge_kind == kind {
                            tracing::trace!(
                                "rewired reverse {} edge from {:?}->{:?} to {:?}->{:?}",
                                edge_kind,
                                input,
                                old_port,
                                input,
                                rewire_to,
                            );

                            *output = rewire_to;
                        }
                    }
                } else {
                    tracing::trace!(
                        "failed to get reverse edge for {:?} while rewiring from {:?} to {:?}",
                        input,
                        old_port,
                        rewire_to,
                    );
                }
            }

            match self.forward.entry(rewire_to) {
                Entry::Occupied(mut forward) => {
                    tracing::trace!(
                        current_edges = ?forward.get(),
                        new_edges = ?forward_edges,
                        "found {} forward edges for {:?}, adding {} new ones",
                        forward.get().len(),
                        rewire_to,
                        forward_edges.len(),
                    );

                    for &(input, _) in forward.get() {
                        let reverse = self.reverse.get_mut(&input).unwrap();

                        for (output, _) in reverse {
                            if *output == old_port {
                                *output = rewire_to;
                            }
                        }
                    }

                    forward.get_mut().reserve(forward_edges.len());
                    for (input, kind) in forward_edges {
                        tracing::trace!(
                            "rewired forward {} edge from {:?}->{:?} to {:?}->{:?}",
                            kind,
                            old_port,
                            input,
                            rewire_to,
                            input,
                        );
                        debug_assert!(
                            !forward.get().contains(&(input, kind)),
                            "edges cannot contain themselves",
                        );

                        forward.get_mut().push((input, kind));
                    }

                    tracing::trace!(forward_edges = ?forward.get());
                }

                Entry::Vacant(vacant) => {
                    tracing::trace!(
                        forward_edges = ?forward_edges,
                        "no forward edges found for {:?}, adding {} new ones",
                        rewire_to,
                        forward_edges.len(),
                    );
                    vacant.insert(forward_edges);
                }
            };
        }

        // Remove any edges that were lying around
        self.remove_output_edges(old_port);
    }

    #[track_caller]
    #[tracing::instrument(skip(self))]
    pub fn splice_ports(&mut self, input: InputPort, output: OutputPort) {
        debug_assert_eq!(self.ports[&input.port()].kind, PortKind::Input);
        debug_assert_eq!(self.ports[&output.port()].kind, PortKind::Output);
        debug_assert_eq!(self.port_parent(input), self.port_parent(output));

        // TODO: Remove allocation
        let sources: TinyVec<[_; 5]> = self
            .inputs_inner(input)
            .map(|(.., source, kind)| (source, kind))
            .collect();
        let consumers: TinyVec<[_; 5]> = self
            .outputs_inner(output)
            .map(|(_, target, ..)| target)
            .collect();

        for (source, kind) in sources {
            for &consumer in &consumers {
                tracing::trace!(
                    "splicing {} edge {:?}->{:?} into {:?}->{:?}",
                    kind,
                    input,
                    output,
                    source,
                    consumer,
                );

                self.add_edge(source, consumer, kind);
            }
        }

        tracing::trace!("removing old ports {:?}, {:?}", input, output);
        self.remove_input_edges(input);
        self.remove_output_edges(output);
    }

    #[track_caller]
    fn assert_value_port<P>(&self, port: P)
    where
        P: Port,
    {
        debug_assert!(self.ports.contains_key(&port.port()));
        debug_assert_eq!(self.ports[&port.port()].edge, EdgeKind::Value);
    }

    #[track_caller]
    fn assert_effect_port<P>(&self, port: P)
    where
        P: Port,
    {
        debug_assert!(self.ports.contains_key(&port.port()));
        debug_assert_eq!(self.ports[&port.port()].edge, EdgeKind::Effect);
    }

    pub fn start(&mut self) -> Start {
        let start_id = self.next_node();

        let effect = self.output_port(start_id, EdgeKind::Effect);

        let start = Start::new(start_id, effect);
        self.nodes
            .insert(start_id, Node::Start(start))
            .debug_unwrap_none();

        start
    }

    #[track_caller]
    pub fn end(&mut self, effect: OutputPort) -> End {
        self.assert_effect_port(effect);

        let end_id = self.next_node();

        let effect_port = self.input_port(end_id, EdgeKind::Effect);
        self.add_effect_edge(effect, effect_port);

        let end = End::new(end_id, effect_port);
        self.nodes
            .insert(end_id, Node::End(end))
            .debug_unwrap_none();

        end
    }

    pub fn int(&mut self, value: i32) -> Int {
        let int_id = self.next_node();

        let output = self.output_port(int_id, EdgeKind::Value);

        let int = Int::new(int_id, output);
        self.nodes
            .insert(int_id, Node::Int(int, value))
            .debug_unwrap_none();

        int
    }

    pub fn bool(&mut self, value: bool) -> Bool {
        let bool_id = self.next_node();

        let output = self.output_port(bool_id, EdgeKind::Value);

        let bool = Bool::new(bool_id, output);
        self.nodes
            .insert(bool_id, Node::Bool(bool, value))
            .debug_unwrap_none();

        bool
    }

    #[track_caller]
    pub fn add(&mut self, lhs: OutputPort, rhs: OutputPort) -> Add {
        self.assert_value_port(lhs);
        self.assert_value_port(rhs);

        let add_id = self.next_node();

        let lhs_port = self.input_port(add_id, EdgeKind::Value);
        self.add_value_edge(lhs, lhs_port);

        let rhs_port = self.input_port(add_id, EdgeKind::Value);
        self.add_value_edge(rhs, rhs_port);

        let output = self.output_port(add_id, EdgeKind::Value);

        let add = Add::new(add_id, lhs_port, rhs_port, output);
        self.nodes
            .insert(add_id, Node::Add(add))
            .debug_unwrap_none();

        add
    }

    #[track_caller]
    pub fn load(&mut self, ptr: OutputPort, effect: OutputPort) -> Load {
        self.assert_value_port(ptr);
        self.assert_effect_port(effect);

        let load_id = self.next_node();

        let ptr_port = self.input_port(load_id, EdgeKind::Value);
        self.add_value_edge(ptr, ptr_port);

        let effect_port = self.input_port(load_id, EdgeKind::Effect);
        self.add_effect_edge(effect, effect_port);

        let loaded = self.output_port(load_id, EdgeKind::Value);
        let effect_out = self.output_port(load_id, EdgeKind::Effect);

        let load = Load::new(load_id, ptr_port, effect_port, loaded, effect_out);
        self.nodes
            .insert(load_id, Node::Load(load))
            .debug_unwrap_none();

        load
    }

    #[track_caller]
    pub fn store(&mut self, ptr: OutputPort, value: OutputPort, effect: OutputPort) -> Store {
        self.assert_value_port(ptr);
        self.assert_value_port(value);
        self.assert_effect_port(effect);

        let store_id = self.next_node();

        let ptr_port = self.input_port(store_id, EdgeKind::Value);
        self.add_value_edge(ptr, ptr_port);

        let value_port = self.input_port(store_id, EdgeKind::Value);
        self.add_value_edge(value, value_port);

        let effect_port = self.input_port(store_id, EdgeKind::Effect);
        self.add_effect_edge(effect, effect_port);

        let effect_out = self.output_port(store_id, EdgeKind::Effect);

        let store = Store::new(store_id, ptr_port, value_port, effect_port, effect_out);
        self.nodes
            .insert(store_id, Node::Store(store))
            .debug_unwrap_none();

        store
    }

    #[track_caller]
    pub fn input(&mut self, effect: OutputPort) -> Input {
        self.assert_effect_port(effect);

        let input_id = self.next_node();

        let effect_port = self.input_port(input_id, EdgeKind::Effect);
        self.add_effect_edge(effect, effect_port);

        let value = self.output_port(input_id, EdgeKind::Value);
        let effect_out = self.output_port(input_id, EdgeKind::Effect);

        let input = Input::new(input_id, effect_port, value, effect_out);
        self.nodes
            .insert(input_id, Node::Input(input))
            .debug_unwrap_none();

        input
    }

    #[track_caller]
    pub fn output(&mut self, value: OutputPort, effect: OutputPort) -> Output {
        self.assert_value_port(value);
        self.assert_effect_port(effect);

        let output_id = self.next_node();

        let value_port = self.input_port(output_id, EdgeKind::Value);
        self.add_value_edge(value, value_port);

        let effect_port = self.input_port(output_id, EdgeKind::Effect);
        self.add_effect_edge(effect, effect_port);

        let effect_out = self.output_port(output_id, EdgeKind::Effect);

        let output = Output::new(output_id, value_port, effect_port, effect_out);
        self.nodes
            .insert(output_id, Node::Output(output))
            .debug_unwrap_none();

        output
    }

    fn input_param(&mut self, kind: EdgeKind) -> InputParam {
        let input_id = self.next_node();

        let port = self.output_port(input_id, EdgeKind::Value);
        let param = InputParam::new(input_id, port, kind);
        self.nodes
            .insert(input_id, Node::InputPort(param))
            .debug_unwrap_none();

        param
    }

    fn output_param(&mut self, input: OutputPort, kind: EdgeKind) -> OutputParam {
        let output_id = self.next_node();

        let port = self.input_port(output_id, EdgeKind::Value);
        self.add_edge(input, port, kind);

        let param = OutputParam::new(output_id, port, kind);
        self.nodes
            .insert(output_id, Node::OutputPort(param))
            .debug_unwrap_none();

        param
    }

    /// Builds a theta node
    ///
    /// `invariant_inputs` is a list of values to be given to the theta's body that
    /// *never change upon iteration*. These values will always stay the same, no matter
    /// how many times the loop iterates.
    ///
    /// `variant_inputs` is a list of values that can change upon iteration. These values
    /// are allowed to evolve as the loop iterates and are fed back into by the `outputs`
    /// field of the [`ThetaData`] constructed in the `build_theta` function.
    ///
    /// `effect` is an optional effect edge to be fed into the theta node. Thetas don't
    /// have to have an input effect, so this is optional.
    ///
    /// The `build_theta` function receives a mutable reference to the theta's body,
    /// the [`OutputPort`] from the body's start node, the [`OutputPort`]s
    /// of all invariant inputs and the [`OutputPort`]s of all variant inputs.
    /// The user doesn't need to create [`Start`] or [`End`] nodes for the theta's
    /// body, these are handled automatically. While it may seem odd that [`ThetaData`]
    /// always requires an [`OutputPort`] for the outgoing effect, this is because
    /// the theta's body still has effect edge flow regardless of whether or not the
    /// outer theta has a incoming/outgoing effects. However, if the outer theta has
    /// no incoming or outgoing effect edges the theta's body effects should be a
    /// direct connection between the body's [`Start`] and [`End`] nodes. That is,
    /// the [`OutputPort`] passed out of the `build_theta` function should be the same
    /// one that was passed to the `build_theta` function as the effect parameter.
    /// Finally, the `condition` of the [`ThetaData`] should be an expression that
    /// evaluates to a boolean value, this is the exit condition of the theta node.
    ///
    /// **The ordering of `invariant_inputs`, `variant_inputs` and `outputs` on the produced
    /// `ThetaData` are all important!!!**
    /// The order of these collections are used to associate things together, the nth element
    /// of the `variant_inputs` parameter will be associated with nth elements of both the
    /// `variant_inputs` slice given to the `build_theta` function and the `outputs` field
    /// of the produced `ThetaData`!
    ///
    pub fn theta<I1, I2, E, F>(
        &mut self,
        invariant_inputs: I1,
        variant_inputs: I2,
        effect: E,
        build_theta: F,
    ) -> ThetaStub
    where
        I1: IntoIterator<Item = OutputPort>,
        I2: IntoIterator<Item = OutputPort>,
        E: Into<Option<OutputPort>>,
        F: FnOnce(&mut Rvsdg, OutputPort, &[OutputPort], &[OutputPort]) -> ThetaData,
    {
        // Create the theta's node id
        let theta_id = self.next_node();

        // If an input effect was given, create a port for it
        let effect_source = effect.into();
        let effect_input = effect_source.map(|effect_source| {
            self.assert_effect_port(effect_source);

            let effect_input = self.input_port(theta_id, EdgeKind::Effect);
            self.add_effect_edge(effect_source, effect_input);

            effect_input
        });

        // Create input ports for the given invariant inputs
        let invariant_input_ports: TinyVec<[_; 5]> = invariant_inputs
            .into_iter()
            .map(|input| {
                self.assert_value_port(input);

                let port = self.input_port(theta_id, EdgeKind::Value);
                self.add_value_edge(input, port);

                port
            })
            .collect();

        // Create input ports for the given variant inputs
        let variant_input_ports: TinyVec<[_; 5]> = variant_inputs
            .into_iter()
            .map(|input| {
                self.assert_value_port(input);

                let port = self.input_port(theta_id, EdgeKind::Value);
                self.add_value_edge(input, port);

                port
            })
            .collect();

        // Create the theta's subgraph
        let mut subgraph =
            Rvsdg::from_counters(self.node_counter.clone(), self.port_counter.clone());

        // Create the theta start node
        let start = subgraph.start();

        // Create the input params for the invariant inputs
        let (invariant_inputs, invariant_param_outputs): (BTreeMap<_, _>, TinyVec<[_; 5]>) =
            invariant_input_ports
                .iter()
                .map(|&input| {
                    let param = subgraph.input_param(EdgeKind::Value);
                    ((input, param.node()), param.output())
                })
                .unzip();

        // Create the input params for the variant inputs
        let (variant_inputs, variant_param_outputs): (BTreeMap<_, _>, TinyVec<[_; 5]>) =
            variant_input_ports
                .iter()
                .map(|&input| {
                    let param = subgraph.input_param(EdgeKind::Value);
                    ((input, param.node()), param.output())
                })
                .unzip();

        // Build the theta node's body
        let ThetaData {
            outputs,
            condition,
            effect: body_effect_output,
        } = build_theta(
            &mut subgraph,
            start.effect(),
            &invariant_param_outputs,
            &variant_param_outputs,
        );

        // Create the subgraph condition's output param
        subgraph.assert_value_port(condition);
        let condition_param = subgraph.output_param(condition, EdgeKind::Value);

        // Create the subgraph's end node
        subgraph.assert_effect_port(body_effect_output);
        let end = subgraph.end(body_effect_output);

        // If there's no input effect then the body can't contain effectful operations
        if effect_input.is_none() {
            assert_eq!(
                body_effect_output,
                start.effect(),
                "if the theta node isn't connected to effect flow, \
                the body cannot have effectful operations",
            );
        }

        // Make sure every variant input has a paired output
        assert_eq!(
            variant_inputs.len(),
            outputs.len(),
            "theta nodes must have the same number of outputs as there are variant inputs",
        );

        // Create the output params for all outputs from the body
        let output_params =
            outputs
                .iter()
                .zip(variant_inputs.keys())
                .map(|(&output, &variant_input)| {
                    subgraph.assert_value_port(output);

                    // Create the output param within the subgraph
                    let output_param = subgraph.output_param(output, EdgeKind::Value);

                    (variant_input, output_param)
                });

        // Create the map of theta output ports to subgraph input params and
        // the map of back edges between output ports and variant inputs
        let (outputs, output_back_edges): (BTreeMap<_, _>, BTreeMap<_, _>) = output_params
            .map(|(variant_input, output_param)| {
                // Create the output port on the theta node
                let output_port = self.output_port(theta_id, EdgeKind::Value);

                (
                    (output_port, output_param.node()),
                    (output_port, variant_input),
                )
            })
            .unzip();
        let output_ports: TinyVec<[OutputPort; 5]> = outputs.keys().copied().collect();

        // If we were given an input effect then we need to make an output effect as well
        let effects = effect_input.map(|effect_input| {
            let effect_output = self.output_port(theta_id, EdgeKind::Effect);

            ThetaEffects::new(effect_input, effect_output)
        });

        let theta = Theta::new(
            theta_id,
            effects,
            invariant_inputs,
            variant_inputs,
            outputs,
            output_back_edges,
            condition_param.node(),
            Box::new(Subgraph::new(subgraph, start.node(), end.node())),
        );

        let stub = ThetaStub::new(effects.map(|effects| effects.output), output_ports);
        self.nodes
            .insert(theta_id, Node::Theta(Box::new(theta)))
            .debug_unwrap_none();

        stub
    }

    #[track_caller]
    pub fn gamma<I, T, F>(
        &mut self,
        inputs: I,
        effect: OutputPort,
        condition: OutputPort,
        truthy: T,
        falsy: F,
    ) -> GammaStub
    where
        I: IntoIterator<Item = OutputPort>,
        T: FnOnce(&mut Rvsdg, OutputPort, &[OutputPort]) -> GammaData,
        F: FnOnce(&mut Rvsdg, OutputPort, &[OutputPort]) -> GammaData,
    {
        self.assert_effect_port(effect);
        self.assert_value_port(condition);

        let gamma_id = self.next_node();

        let effect_in = self.input_port(gamma_id, EdgeKind::Effect);
        self.add_effect_edge(effect, effect_in);

        let cond_port = self.input_port(gamma_id, EdgeKind::Value);
        self.add_value_edge(condition, cond_port);

        // Wire up the external inputs to the gamma node
        let outer_inputs: TinyVec<[_; 5]> = inputs
            .into_iter()
            .map(|input| {
                self.assert_value_port(input);

                let port = self.input_port(gamma_id, EdgeKind::Value);
                self.add_value_edge(input, port);
                port
            })
            .collect();

        // Create the gamma's true branch
        let mut truthy_subgraph =
            Rvsdg::from_counters(self.node_counter.clone(), self.port_counter.clone());

        // Create the input ports within the subgraph
        let (truthy_input_params, truthy_inner_input_ports): (TinyVec<[_; 5]>, TinyVec<[_; 5]>) =
            (0..outer_inputs.len())
                .map(|_| {
                    let param = truthy_subgraph.input_param(EdgeKind::Value);
                    (param.node, param.output())
                })
                .unzip();

        // Create the branch's start node
        let truthy_start = truthy_subgraph.start();
        let GammaData {
            outputs: truthy_outputs,
            effect: truthy_output_effect,
        } = truthy(
            &mut truthy_subgraph,
            truthy_start.effect(),
            &truthy_inner_input_ports,
        );

        truthy_subgraph.assert_effect_port(truthy_output_effect);
        let truthy_end = truthy_subgraph.end(truthy_output_effect);

        let truthy_output_params: TinyVec<[_; 5]> = truthy_outputs
            .iter()
            .map(|&output| {
                truthy_subgraph.assert_value_port(output);
                truthy_subgraph.output_param(output, EdgeKind::Value).node()
            })
            .collect();

        // Create the gamma's true branch
        let mut falsy_subgraph =
            Rvsdg::from_counters(self.node_counter.clone(), self.port_counter.clone());

        // Create the input ports within the subgraph
        let (falsy_input_params, falsy_inner_input_ports): (TinyVec<[_; 5]>, TinyVec<[_; 5]>) = (0
            ..outer_inputs.len())
            .map(|_| {
                let param = falsy_subgraph.input_param(EdgeKind::Value);
                (param.node, param.output())
            })
            .unzip();

        // Create the branch's start node
        let falsy_start = falsy_subgraph.start();
        let GammaData {
            outputs: falsy_outputs,
            effect: falsy_output_effect,
        } = falsy(
            &mut falsy_subgraph,
            falsy_start.effect(),
            &falsy_inner_input_ports,
        );

        falsy_subgraph.assert_effect_port(falsy_output_effect);
        let falsy_end = falsy_subgraph.end(falsy_output_effect);

        let falsy_output_params: TinyVec<[_; 5]> = falsy_outputs
            .iter()
            .map(|&output| {
                falsy_subgraph.assert_value_port(output);
                falsy_subgraph.output_param(output, EdgeKind::Value).node()
            })
            .collect();

        // FIXME: I'd really like to be able to support variable numbers of inputs for each branch
        //        to allow some more flexible optimizations like removing effect flow from a branch
        debug_assert_eq!(truthy_input_params.len(), falsy_input_params.len());
        // FIXME: Remove the temporary allocations
        let input_params = truthy_input_params
            .into_iter()
            .zip(falsy_input_params)
            .map(|(truthy, falsy)| [truthy, falsy])
            .collect();

        debug_assert_eq!(truthy_output_params.len(), falsy_output_params.len());
        // FIXME: Remove the temporary allocations
        let output_params: TinyVec<[_; 5]> = truthy_output_params
            .into_iter()
            .zip(falsy_output_params)
            .map(|(truthy, falsy)| [truthy, falsy])
            .collect();

        let effect_out = self.output_port(gamma_id, EdgeKind::Effect);
        let outer_outputs: TinyVec<[_; 5]> = (0..output_params.len())
            .map(|_| self.output_port(gamma_id, EdgeKind::Value))
            .collect();

        let stub = GammaStub::new(Some(effect_out), outer_outputs.iter().copied().collect());
        let gamma = Gamma::new(
            gamma_id,                                    // node
            outer_inputs,                                // inputs
            effect_in,                                   // effect_in
            input_params,                                // input_params
            falsy_start.effect(),                        // input_effect
            outer_outputs,                               // outputs
            effect_out,                                  // effect_out
            output_params,                               // output_params
            [truthy_output_effect, falsy_output_effect], // output_effect
            [truthy_start.node, falsy_start.node],       // start_nodes
            [truthy_end.node, falsy_end.node],           // end_nodes
            Box::new([truthy_subgraph, falsy_subgraph]), // body
            cond_port,                                   // condition
        );

        self.nodes
            .insert(gamma_id, Node::Gamma(Box::new(gamma)))
            .debug_unwrap_none();

        stub
    }

    #[track_caller]
    pub fn eq(&mut self, lhs: OutputPort, rhs: OutputPort) -> Eq {
        self.assert_value_port(lhs);
        self.assert_value_port(rhs);

        let eq_id = self.next_node();

        let lhs_port = self.input_port(eq_id, EdgeKind::Value);
        self.add_value_edge(lhs, lhs_port);

        let rhs_port = self.input_port(eq_id, EdgeKind::Value);
        self.add_value_edge(rhs, rhs_port);

        let output = self.output_port(eq_id, EdgeKind::Value);

        let eq = Eq::new(eq_id, lhs_port, rhs_port, output);
        self.nodes.insert(eq_id, Node::Eq(eq)).debug_unwrap_none();

        eq
    }

    #[track_caller]
    pub fn not(&mut self, input: OutputPort) -> Not {
        self.assert_value_port(input);

        let not_id = self.next_node();

        let input_port = self.input_port(not_id, EdgeKind::Value);
        self.add_value_edge(input, input_port);

        let output = self.output_port(not_id, EdgeKind::Value);

        let not = Not::new(not_id, input_port, output);
        self.nodes
            .insert(not_id, Node::Not(not))
            .debug_unwrap_none();

        not
    }

    #[track_caller]
    pub fn neg(&mut self, input: OutputPort) -> Neg {
        self.assert_value_port(input);

        let neg_id = self.next_node();

        let input_port = self.input_port(neg_id, EdgeKind::Value);
        self.add_value_edge(input, input_port);

        let output = self.output_port(neg_id, EdgeKind::Value);

        let neg = Neg::new(neg_id, input_port, output);
        self.nodes
            .insert(neg_id, Node::Neg(neg))
            .debug_unwrap_none();

        neg
    }

    #[track_caller]
    pub fn neq(&mut self, lhs: OutputPort, rhs: OutputPort) -> Not {
        self.assert_value_port(lhs);
        self.assert_value_port(rhs);

        let eq = self.eq(lhs, rhs);
        self.not(eq.value())
    }
}

impl Default for Rvsdg {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PortData {
    kind: PortKind,
    edge: EdgeKind,
    parent: NodeId,
}

impl PortData {
    const fn new(kind: PortKind, edge: EdgeKind, node: NodeId) -> Self {
        Self {
            kind,
            edge,
            parent: node,
        }
    }

    const fn input(node: NodeId, edge: EdgeKind) -> Self {
        Self::new(PortKind::Input, edge, node)
    }

    const fn output(node: NodeId, edge: EdgeKind) -> Self {
        Self::new(PortKind::Output, edge, node)
    }
}

impl Display for PortData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} port for {}", self.kind, self.parent)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PortKind {
    Input,
    Output,
}

impl PortKind {
    /// Returns `true` if the port kind is [`Input`].
    ///
    /// [`Input`]: PortKind::Input
    const fn is_input(&self) -> bool {
        matches!(self, Self::Input)
    }

    /// Returns `true` if the port kind is [`Output`].
    ///
    /// [`Output`]: PortKind::Output
    const fn is_output(&self) -> bool {
        matches!(self, Self::Output)
    }
}

impl Display for PortKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input => f.write_str("input"),
            Self::Output => f.write_str("output"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Effect,
    Value,
}

impl EdgeKind {
    /// Returns `true` if the edge kind is [`Effect`].
    ///
    /// [`Effect`]: EdgeKind::Effect
    pub const fn is_effect(&self) -> bool {
        matches!(self, Self::Effect)
    }

    /// Returns `true` if the edge kind is [`Value`].
    ///
    /// [`Value`]: EdgeKind::Value
    pub const fn is_value(&self) -> bool {
        matches!(self, Self::Value)
    }
}

impl Default for EdgeKind {
    fn default() -> Self {
        Self::Value
    }
}

impl Display for EdgeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Effect => f.write_str("effect"),
            Self::Value => f.write_str("value"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Add {
    node: NodeId,
    lhs: InputPort,
    rhs: InputPort,
    value: OutputPort,
}

impl Add {
    const fn new(node: NodeId, lhs: InputPort, rhs: InputPort, value: OutputPort) -> Self {
        Self {
            node,
            lhs,
            rhs,
            value,
        }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// Get the add's left hand side
    pub const fn lhs(&self) -> InputPort {
        self.lhs
    }

    #[allow(dead_code)]
    pub fn lhs_mut(&mut self) -> &mut InputPort {
        &mut self.lhs
    }

    /// Get the add's right hand side
    pub const fn rhs(&self) -> InputPort {
        self.rhs
    }

    #[allow(dead_code)]
    pub fn rhs_mut(&mut self) -> &mut InputPort {
        &mut self.rhs
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Int {
    node: NodeId,
    value: OutputPort,
}

impl Int {
    pub const fn new(node: NodeId, value: OutputPort) -> Self {
        Self { node, value }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bool {
    node: NodeId,
    value: OutputPort,
}

impl Bool {
    pub const fn new(node: NodeId, value: OutputPort) -> Self {
        Self { node, value }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Load {
    node: NodeId,
    ptr: InputPort,
    effect_in: InputPort,
    value: OutputPort,
    effect_out: OutputPort,
}

impl Load {
    const fn new(
        node: NodeId,
        ptr: InputPort,
        effect_in: InputPort,
        value: OutputPort,
        effect_out: OutputPort,
    ) -> Self {
        Self {
            node,
            ptr,
            effect_in,
            value,
            effect_out,
        }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn ptr(&self) -> InputPort {
        self.ptr
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }

    pub const fn effect_in(&self) -> InputPort {
        self.effect_in
    }

    pub const fn effect(&self) -> OutputPort {
        self.effect_out
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Store {
    node: NodeId,
    ptr: InputPort,
    value: InputPort,
    effect_in: InputPort,
    effect_out: OutputPort,
}

impl Store {
    const fn new(
        node: NodeId,
        ptr: InputPort,
        value: InputPort,
        effect_in: InputPort,
        effect_out: OutputPort,
    ) -> Self {
        Self {
            node,
            ptr,
            value,
            effect_in,
            effect_out,
        }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn ptr(&self) -> InputPort {
        self.ptr
    }

    pub const fn value(&self) -> InputPort {
        self.value
    }

    #[allow(dead_code)]
    pub fn value_mut(&mut self) -> &mut InputPort {
        &mut self.value
    }

    pub const fn effect_in(&self) -> InputPort {
        self.effect_in
    }

    pub const fn effect(&self) -> OutputPort {
        self.effect_out
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Start {
    node: NodeId,
    effect: OutputPort,
}

impl Start {
    const fn new(node: NodeId, effect: OutputPort) -> Self {
        Self { node, effect }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn effect(&self) -> OutputPort {
        self.effect
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct End {
    node: NodeId,
    effect: InputPort,
}

impl End {
    const fn new(node: NodeId, effect: InputPort) -> Self {
        Self { node, effect }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    #[allow(dead_code)]
    pub const fn effect_in(&self) -> InputPort {
        self.effect
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InputParam {
    node: NodeId,
    output: OutputPort,
    kind: EdgeKind,
}

impl InputParam {
    const fn new(node: NodeId, output: OutputPort, kind: EdgeKind) -> Self {
        Self { node, output, kind }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn output(&self) -> OutputPort {
        self.output
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutputParam {
    node: NodeId,
    input: InputPort,
    kind: EdgeKind,
}

impl OutputParam {
    const fn new(node: NodeId, input: InputPort, kind: EdgeKind) -> Self {
        Self { node, input, kind }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn input(&self) -> InputPort {
        self.input
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Input {
    node: NodeId,
    effect_in: InputPort,
    value: OutputPort,
    effect_out: OutputPort,
}

impl Input {
    const fn new(
        node: NodeId,
        effect_in: InputPort,
        value: OutputPort,
        effect_out: OutputPort,
    ) -> Self {
        Self {
            node,
            effect_in,
            value,
            effect_out,
        }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }

    pub const fn effect_in(&self) -> InputPort {
        self.effect_in
    }

    pub const fn effect(&self) -> OutputPort {
        self.effect_out
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Output {
    node: NodeId,
    value: InputPort,
    effect_in: InputPort,
    effect_out: OutputPort,
}

impl Output {
    const fn new(
        node: NodeId,
        value: InputPort,
        effect_in: InputPort,
        effect_out: OutputPort,
    ) -> Self {
        Self {
            node,
            value,
            effect_in,
            effect_out,
        }
    }

    pub const fn value(&self) -> InputPort {
        self.value
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn effect_in(&self) -> InputPort {
        self.effect_in
    }

    pub const fn effect(&self) -> OutputPort {
        self.effect_out
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eq {
    node: NodeId,
    lhs: InputPort,
    rhs: InputPort,
    value: OutputPort,
}

impl Eq {
    const fn new(node: NodeId, lhs: InputPort, rhs: InputPort, value: OutputPort) -> Self {
        Self {
            node,
            lhs,
            rhs,
            value,
        }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// Get the eq's left hand side
    pub const fn lhs(&self) -> InputPort {
        self.lhs
    }

    /// Get the eq's right hand side
    pub const fn rhs(&self) -> InputPort {
        self.rhs
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Not {
    node: NodeId,
    input: InputPort,
    value: OutputPort,
}

impl Not {
    const fn new(node: NodeId, input: InputPort, value: OutputPort) -> Self {
        Self { node, input, value }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// Get the not's input
    pub const fn input(&self) -> InputPort {
        self.input
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neg {
    node: NodeId,
    input: InputPort,
    value: OutputPort,
}

impl Neg {
    const fn new(node: NodeId, input: InputPort, value: OutputPort) -> Self {
        Self { node, input, value }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// Get the not's input
    pub const fn input(&self) -> InputPort {
        self.input
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct NodeId(pub u32);

impl NodeId {
    const fn new(id: u32) -> Self {
        Self(id)
    }
}

impl Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NodeId(")?;
        Debug::fmt(&self.0, f)?;
        f.write_char(')')
    }
}

impl Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}
