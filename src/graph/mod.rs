mod building;
mod edge;
mod nodes;
mod ports;
mod remove;
mod stats;
mod subgraph;

pub use edge::{EdgeCount, EdgeDescriptor, EdgeKind};
pub use nodes::{
    Add, Bool, Byte, End, Eq, Gamma, GammaData, Input, InputParam, Int, Load, Mul, Neg, Node,
    NodeExt, NodeId, Not, Output, OutputParam, Start, Store, Sub, Theta, ThetaData,
};
pub use ports::{InputPort, OutputPort, Port, PortData, PortId, PortKind};
pub use subgraph::Subgraph;

use crate::utils::AssertNone;
use std::{
    cell::Cell,
    collections::{btree_map::Entry, BTreeMap},
    fmt::Debug,
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
    start_nodes: TinyVec<[NodeId; 1]>,
    end_nodes: TinyVec<[NodeId; 1]>,
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
            start_nodes: TinyVec::new(),
            end_nodes: TinyVec::new(),
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
    pub fn node_len(&self) -> usize {
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

    /// Runs the provided closure for each node within the graph and all contained subgraphs
    pub fn for_each_transitive_node<F>(&self, mut for_each: F)
    where
        F: FnMut(NodeId, &Node),
    {
        self.for_each_transitive_node_inner(&mut for_each);
    }

    pub fn for_each_transitive_node_inner<F>(&self, for_each: &mut F)
    where
        F: FnMut(NodeId, &Node),
    {
        for (&node_id, node) in self.nodes.iter() {
            for_each(node_id, node);

            match node {
                Node::Gamma(gamma) => {
                    gamma.true_branch().for_each_transitive_node_inner(for_each);
                    gamma
                        .false_branch()
                        .for_each_transitive_node_inner(for_each);
                }
                Node::Theta(theta) => theta.body().for_each_transitive_node_inner(for_each),

                _ => {}
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
        self.ports
            .get(&port.port())
            .expect("failed to find parent for port")
            .parent
    }

    pub fn incoming_count(&self, node: NodeId) -> usize {
        self.get_node(node)
            .all_input_ports()
            .into_iter()
            .flat_map(|input| self.inputs_inner(input).map(|(.., count)| count as usize))
            .sum()
    }

    pub fn value_input_count(&self, node: NodeId) -> usize {
        self.get_node(node)
            .all_input_ports()
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
            .all_input_ports()
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
            .all_output_ports()
            .into_iter()
            .flat_map(|output| self.outputs_inner(output).map(|(.., count)| count as usize))
            .sum()
    }

    pub fn value_output_count(&self, node: NodeId) -> usize {
        self.get_node(node)
            .all_output_ports()
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
            .all_output_ports()
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

    #[inline]
    #[track_caller]
    pub fn input_source_node(&self, input: InputPort) -> &Node {
        self.get_input(input).0
    }

    #[inline]
    #[track_caller]
    pub fn cast_input_source<T>(&self, input: InputPort) -> Option<&T>
    where
        for<'a> &'a Node: TryInto<&'a T>,
    {
        self.input_source_node(input).try_into().ok()
    }

    #[inline]
    #[track_caller]
    pub fn input_source_id(&self, input: InputPort) -> NodeId {
        self.input_source_node(input).node()
    }

    #[inline]
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
    pub fn to_node<T>(&self, node: NodeId) -> &T
    where
        for<'a> &'a Node: TryInto<&'a T>,
    {
        if let Some(node) = self.cast_node::<T>(node) {
            node
        } else {
            panic!(
                "failed to cast {:?} to {}: {:?}",
                node,
                std::any::type_name::<T>(),
                self.try_node(node),
            )
        }
    }

    pub fn cast_node<T>(&self, node: NodeId) -> Option<&T>
    where
        for<'a> &'a Node: TryInto<&'a T>,
    {
        self.try_node(node).and_then(|node| node.try_into().ok())
    }

    pub fn cast_source<T>(&self, target: InputPort) -> Option<&T>
    where
        for<'a> &'a Node: TryInto<&'a T>,
    {
        self.try_input(target)
            .and_then(|(node, _, _)| node.try_into().ok())
    }

    pub fn cast_target<T>(&self, source: OutputPort) -> Option<&T>
    where
        for<'a> &'a Node: TryInto<&'a T>,
    {
        self.get_output(source)
            .and_then(|(node, _, _)| node.try_into().ok())
    }

    pub fn cast_parent<P, T>(&self, port: P) -> Option<&T>
    where
        P: Port,
        for<'a> &'a Node: TryInto<&'a T>,
    {
        self.cast_node(self.port_parent(port))
    }

    #[inline]
    #[track_caller]
    pub fn get_output(&self, output: OutputPort) -> Option<(&Node, InputPort, EdgeKind)> {
        self.get_outputs(output).next()
    }

    #[inline]
    #[track_caller]
    pub fn output_dest_node(&self, output: OutputPort) -> Option<&Node> {
        self.get_output(output).map(|(node, ..)| node)
    }

    #[inline]
    #[track_caller]
    pub fn output_dest_id(&self, output: OutputPort) -> Option<NodeId> {
        self.output_dest_node(output).map(NodeExt::node)
    }

    #[inline]
    #[track_caller]
    pub fn cast_output_dest<T>(&self, output: OutputPort) -> Option<&T>
    where
        for<'a> &'a Node: TryInto<&'a T>,
    {
        self.output_dest_node(output)
            .and_then(|node| node.try_into().ok())
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
    pub fn all_node_inputs(
        &self,
        node: NodeId,
    ) -> impl Iterator<Item = (InputPort, &Node, OutputPort, EdgeKind)> + '_ {
        let node = self.get_node(node);

        // TODO: Remove need for `.inputs()` call & allocation here
        node.all_input_ports().into_iter().flat_map(|input| {
            self.inputs_inner(input)
                .map(|(node, input, output, edge)| (input, node, output, edge))
        })
    }

    #[track_caller]
    pub fn all_node_input_sources(&self, node: NodeId) -> impl Iterator<Item = OutputPort> + '_ {
        self.all_node_inputs(node).map(|(.., source, _)| source)
    }

    #[track_caller]
    pub fn all_node_input_source_ids(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.all_node_input_sources(node)
            .map(|source| self.port_parent(source))
    }

    #[allow(dead_code)]
    pub fn try_inputs(
        &self,
        node: NodeId,
    ) -> impl Iterator<Item = (InputPort, Option<(&Node, OutputPort, EdgeKind)>)> {
        let node = self.get_node(node);

        node.all_input_ports()
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
            .all_output_ports()
            .into_iter()
            .map(|output| (output, self.get_output(output)))
    }

    // FIXME: Invariant assertions and logging
    #[tracing::instrument(level = "trace", skip(self))]
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
    #[tracing::instrument(level = "trace", skip(self))]
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

    #[tracing::instrument(level = "trace", skip(self))]
    pub fn remove_input_port(&mut self, input: InputPort) {
        tracing::trace!(
            "removing input port {:?} from {:?}",
            input,
            self.port_parent(input),
        );

        self.remove_input_edges(input);
        self.ports.remove(&input.port());
    }

    #[tracing::instrument(level = "trace", skip(self))]
    pub fn remove_output_port(&mut self, output: OutputPort) {
        tracing::trace!(
            "removing output port {:?} from {:?}",
            output,
            self.port_parent(output),
        );

        self.remove_output_edges(output);
        self.ports.remove(&output.port());
    }

    #[tracing::instrument(level = "trace", skip(self))]
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

    #[tracing::instrument(level = "trace", skip(self))]
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

    // FIXME: Invariant assertions & logging
    #[track_caller]
    #[tracing::instrument(level = "trace", skip(self))]
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

                    // for &(input, kind) in forward.get() {
                    //     let reverse = self.reverse.get_mut(&input).unwrap();
                    //
                    //     for (output, edge_kind) in reverse {
                    //         if *output == old_port && *edge_kind == kind {
                    //             tracing::trace!(
                    //                 "rewired reverse {} edge from {:?}->{:?} to {:?}->{:?}",
                    //                 edge_kind,
                    //                 input,
                    //                 old_port,
                    //                 input,
                    //                 rewire_to,
                    //             );
                    //
                    //             *output = rewire_to;
                    //         }
                    //     }
                    // }

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
    #[tracing::instrument(level = "trace", skip(self))]
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

    /// Get a reference to the rvsdg's start nodes.
    pub fn start_nodes(&self) -> &[NodeId] {
        &self.start_nodes
    }

    /// Get a reference to the rvsdg's end nodes.
    pub fn end_nodes(&self) -> &[NodeId] {
        &self.end_nodes
    }
}

impl Default for Rvsdg {
    fn default() -> Self {
        Self::new()
    }
}
