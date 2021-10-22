mod node;
mod stats;

pub use node::Node;

use petgraph::{
    dot::Dot,
    stable_graph::{NodeIndex, StableGraph},
    visit::EdgeRef,
    Direction,
};
use std::{
    cell::Cell,
    cmp,
    collections::{BTreeMap, BTreeSet},
    fmt::{self, Debug, Display, Write as _},
    fs::{self, File},
    hash::Hash,
    panic::Location,
    path::Path,
    rc::Rc,
};

#[derive(Debug, Clone)]
pub struct Rvsdg {
    graph: StableGraph<PortData, EdgeKind>,
    nodes: BTreeMap<NodeId, Node>,
    counter: Rc<Cell<u32>>,
}

impl Rvsdg {
    pub fn new() -> Self {
        Self::from_counter(Rc::new(Cell::new(0)))
    }

    fn from_counter(counter: Rc<Cell<u32>>) -> Self {
        Self {
            graph: StableGraph::new(),
            nodes: BTreeMap::new(),
            counter,
        }
    }

    #[allow(dead_code)]
    pub fn to_dot(&self, path: impl AsRef<Path>) {
        use std::io::Write;

        let path = path.as_ref();
        let dot = Dot::new(&self.graph);

        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut file = File::create(path).unwrap();

        write!(&mut file, "{}", dot).unwrap();
    }

    fn next_node(&mut self) -> NodeId {
        let node = NodeId(self.counter.get());
        self.counter.set(node.0 + 1);

        node
    }

    #[track_caller]
    pub fn add_edge(&mut self, src: OutputPort, dest: InputPort, kind: EdgeKind) {
        // TODO: Add some more invariant assertions here, ideally for
        //       ensuring that we're creating the expected edge type for
        //       the given ports
        debug_assert_eq!(self.graph[src.0].kind, PortKind::Output);
        debug_assert_eq!(self.graph[dest.0].kind, PortKind::Input);

        self.graph.add_edge(src.0, dest.0, kind);
    }

    #[allow(dead_code)]
    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn contains_node(&self, node: NodeId) -> bool {
        self.nodes.contains_key(&node)
    }

    pub fn add_node(&mut self, node_id: NodeId, node: Node) {
        let displaced = self.nodes.insert(node_id, node);
        debug_assert!(displaced.is_none());
    }

    #[track_caller]
    pub fn add_value_edge(&mut self, src: OutputPort, dest: InputPort) {
        self.add_edge(src, dest, EdgeKind::Value);
    }

    #[track_caller]
    pub fn add_effect_edge(&mut self, src: OutputPort, dest: InputPort) {
        self.add_edge(src, dest, EdgeKind::Effect);
    }

    pub fn input_port(&mut self, parent: NodeId, edge: EdgeKind) -> InputPort {
        InputPort(self.graph.add_node(PortData::input(parent, edge)))
    }

    pub fn output_port(&mut self, parent: NodeId, edge: EdgeKind) -> OutputPort {
        OutputPort(self.graph.add_node(PortData::output(parent, edge)))
    }

    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.keys().copied()
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
        self.graph[port.index()].parent
    }

    pub fn incoming_count(&self, node: NodeId) -> usize {
        self.get_node(node)
            .inputs()
            .into_iter()
            .map(|input| {
                self.graph
                    .edges_directed(input.0, Direction::Incoming)
                    .filter(|edge| self.nodes.contains_key(&self.graph[edge.source()].parent))
                    .count()
            })
            .sum()
    }

    pub fn value_input_count(&self, node: NodeId) -> usize {
        self.get_node(node)
            .inputs()
            .into_iter()
            .map(|input| {
                self.graph
                    .edges_directed(input.0, Direction::Incoming)
                    .filter(|edge| {
                        *edge.weight() == EdgeKind::Value
                            && self.nodes.contains_key(&self.graph[edge.source()].parent)
                    })
                    .count()
            })
            .sum()
    }

    pub fn effect_input_count(&self, node: NodeId) -> usize {
        self.get_node(node)
            .inputs()
            .into_iter()
            .map(|input| {
                self.graph
                    .edges_directed(input.0, Direction::Incoming)
                    .filter(|edge| {
                        *edge.weight() == EdgeKind::Effect
                            && self.nodes.contains_key(&self.graph[edge.source()].parent)
                    })
                    .count()
            })
            .sum()
    }

    pub fn outgoing_count(&self, node: NodeId) -> usize {
        self.get_node(node)
            .outputs()
            .into_iter()
            .map(move |output| {
                self.graph
                    .edges_directed(output.0, Direction::Outgoing)
                    .filter(|edge| self.nodes.contains_key(&self.graph[edge.target()].parent))
                    .count()
            })
            .sum()
    }

    pub fn value_output_count(&self, node: NodeId) -> usize {
        self.get_node(node)
            .outputs()
            .into_iter()
            .map(move |output| {
                self.graph
                    .edges_directed(output.0, Direction::Outgoing)
                    .filter(|edge| {
                        *edge.weight() == EdgeKind::Value
                            && self.nodes.contains_key(&self.graph[edge.target()].parent)
                    })
                    .count()
            })
            .sum()
    }

    pub fn effect_output_count(&self, node: NodeId) -> usize {
        self.get_node(node)
            .outputs()
            .into_iter()
            .map(move |output| {
                self.graph
                    .edges_directed(output.0, Direction::Outgoing)
                    .filter(|edge| {
                        *edge.weight() == EdgeKind::Effect
                            && self.nodes.contains_key(&self.graph[edge.target()].parent)
                    })
                    .count()
            })
            .sum()
    }

    pub fn input_source(&self, input: InputPort) -> OutputPort {
        let mut incoming = self.graph.edges_directed(input.0, Direction::Incoming);
        debug_assert_eq!(incoming.clone().count(), 1);

        let edge = incoming.next().unwrap();
        debug_assert_eq!(self.graph[edge.source()].kind, PortKind::Output);

        OutputPort(edge.source())
    }

    pub fn output_dest(&self, output: OutputPort) -> Option<InputPort> {
        let mut outgoing = self.graph.edges_directed(output.0, Direction::Outgoing);
        // FIXME: `debug_assert_matches!()`
        debug_assert!(matches!(outgoing.clone().count(), 0 | 1));

        outgoing.next().map(|edge| {
            debug_assert_eq!(self.graph[edge.target()].kind, PortKind::Input);
            InputPort(edge.target())
        })
    }

    #[track_caller]
    pub fn get_input(&self, input: InputPort) -> (&Node, OutputPort, EdgeKind) {
        match self.try_input(input) {
            Some(input) => input,
            None => {
                let port = self.graph[input.0];

                panic!(
                    "incorrect number of edges found for input port {:?} \
                     (port data: {:?}, node: {:?})",
                    input, port, self.nodes[&port.parent],
                );
            }
        }
    }

    pub fn try_input(&self, input: InputPort) -> Option<(&Node, OutputPort, EdgeKind)> {
        if self.graph.contains_node(input.0) {
            let port = self.graph[input.0];
            debug_assert_eq!(port.kind, PortKind::Input);

            let mut incoming = self.graph.edges_directed(input.0, Direction::Incoming);
            // FIXME: debug_assert_matches!()
            // debug_assert!(
            //     matches!(incoming.clone().count(), 0 | 1),
            //     "incorrect number of edges found for input port {:?}, \
            //      expected 0 or 1 but got {} (port data: {:?}, node: {:?})",
            //     input,
            //     incoming.clone().count(),
            //     port,
            //     self.nodes[&port.parent],
            // );

            incoming.find_map(|edge| {
                let (src, kind) = (edge.source(), *edge.weight());
                debug_assert_eq!(self.graph[src].kind, PortKind::Output);

                self.nodes
                    .get(&self.graph[src].parent)
                    .map(|node| (node, OutputPort(src), kind))
            })
        } else {
            None
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
        if cfg!(debug_assertions) && self.graph.contains_node(output.0) {
            let port = self.graph[output.0];
            debug_assert_eq!(port.kind, PortKind::Output);
        }

        self.graph
            .edges_directed(output.0, Direction::Outgoing)
            .next()
            .map(|edge| {
                let dest = edge.target();
                let node = &self.nodes[&self.graph[dest].parent];
                debug_assert_eq!(self.graph[edge.target()].kind, PortKind::Input);

                (node, InputPort(edge.target()), *edge.weight())
            })
    }

    pub fn get_outputs(
        &self,
        output: OutputPort,
    ) -> impl Iterator<Item = (&Node, InputPort, EdgeKind)> {
        self.graph
            .edges_directed(output.0, Direction::Outgoing)
            .map(|edge| {
                let dest = edge.target();
                let node = &self.nodes[&self.graph[dest].parent];
                debug_assert_eq!(self.graph[edge.target()].kind, PortKind::Input);

                (node, InputPort(edge.target()), *edge.weight())
            })
    }

    #[track_caller]
    pub fn inputs(
        &self,
        node: NodeId,
    ) -> impl Iterator<Item = (InputPort, &Node, OutputPort, EdgeKind)> {
        let caller = Location::caller();
        let node = self.get_node(node);

        node.inputs().into_iter().map(move |input| {
            let port = self.graph[input.0];
            let (node, output, kind) = self.try_input(input).unwrap_or_else(|| {
                panic!(
                    "missing edge for input port {:?} (port data: {:?}, node: {:?}) @ {}:{}:{}",
                    input,
                    port,
                    self.nodes[&port.parent],
                    caller.file(),
                    caller.line(),
                    caller.column(),
                )
            });

            (input, node, output, kind)
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
    pub fn outputs(
        &self,
        node: NodeId,
    ) -> impl Iterator<Item = (OutputPort, Option<(&Node, InputPort, EdgeKind)>)> {
        self.get_node(node)
            .outputs()
            .into_iter()
            .map(|output| (output, self.get_output(output)))
    }

    pub fn remove_input_edge(&mut self, input: InputPort) {
        debug_assert_eq!(self.graph[input.0].kind, PortKind::Input);

        let mut incoming = self.graph.edges_directed(input.0, Direction::Incoming);
        // debug_assert_eq!(incoming.clone().count(), 1);

        if let Some(edge) = incoming.next().map(|edge| edge.id()) {
            self.graph.remove_edge(edge);
        }
    }

    pub fn remove_output_edge(&mut self, output: OutputPort) {
        let outgoing = self
            .graph
            .edges_directed(output.0, Direction::Outgoing)
            .next()
            .map(|edge| edge.id());

        if let Some(edge) = outgoing {
            tracing::trace!("removing output port's edge {} (edge: {:?})", output, edge);
            debug_assert_eq!(self.graph[output.0].kind, PortKind::Output);

            self.graph.remove_edge(edge);
        } else {
            tracing::trace!("output {} doesn't exist, not removing edge", output);
        }
    }

    pub fn remove_output_edges(&mut self, output: OutputPort) {
        if cfg!(debug_assertions) && self.graph.contains_node(output.0) {
            debug_assert_eq!(self.graph[output.0].kind, PortKind::Output);
        }

        self.graph.retain_edges(|graph, edge| {
            if let Some((source, target)) = graph.edge_endpoints(edge) {
                if source == output.0 {
                    tracing::trace!(
                        "removing output port's edge {} (edge: {:?} from {:?}->{:?})",
                        output,
                        edge,
                        output,
                        InputPort(target),
                    );

                    return true;
                }
            }

            false
        });
    }

    pub fn remove_output_port(&mut self, output: OutputPort) {
        self.remove_output_edge(output);
        self.graph.remove_node(output.0);
    }

    pub fn remove_inputs(&mut self, node: NodeId) {
        for input in self.get_node(node).inputs() {
            self.graph.remove_node(input.0);
        }
    }

    pub fn remove_outputs(&mut self, node: NodeId) {
        for output in self.get_node(node).outputs() {
            self.graph.remove_node(output.0);
        }
    }

    pub fn replace_node<N: Into<Node>>(&mut self, node_id: NodeId, node: N) {
        *self
            .nodes
            .get_mut(&node_id)
            .expect("attempted to replace a node that doesn't exist") = node.into();
    }

    pub fn remove_node(&mut self, node: NodeId) {
        self.remove_inputs(node);
        self.remove_outputs(node);
        self.nodes.remove(&node);
    }

    #[track_caller]
    pub fn rewire_dependents(&mut self, old_port: OutputPort, rewire_to: OutputPort) {
        debug_assert_eq!(self.graph[old_port.0].kind, PortKind::Output);
        debug_assert_eq!(self.graph[rewire_to.0].kind, PortKind::Output);

        // FIXME: Remove this allocation
        let edges: Vec<_> = self
            .graph
            .edges_directed(old_port.0, Direction::Outgoing)
            .map(|edge| (edge.id(), edge.target(), *edge.weight()))
            .collect();

        if edges.is_empty() {
            tracing::debug!(
                "tried to rewire dependents from {:?} to {:?}, but none were found",
                old_port,
                rewire_to,
            );
        } else {
            for (edge, dest, kind) in edges {
                tracing::trace!(
                    "rewiring {:?}->{:?} into {:?}->{:?}",
                    old_port,
                    InputPort(dest),
                    rewire_to,
                    InputPort(dest),
                );
                debug_assert_eq!(self.graph[dest].kind, PortKind::Input);

                self.graph.remove_edge(edge);
                self.graph.add_edge(rewire_to.0, dest, kind);
            }
        }
    }

    #[track_caller]
    pub fn splice_ports(&mut self, input: InputPort, output: OutputPort) {
        debug_assert_eq!(self.graph[input.0].kind, PortKind::Input);
        debug_assert_eq!(self.graph[output.0].kind, PortKind::Output);
        debug_assert_eq!(self.port_parent(input), self.port_parent(output));

        let mut incoming = self.graph.edges_directed(input.0, Direction::Incoming);
        // FIXME: `debug_assert_matches!()`
        debug_assert!(matches!(incoming.clone().count(), 1 | 0));

        if let Some((src, src_kind)) = incoming.next().map(|edge| (edge.source(), *edge.weight())) {
            for (dest, dest_kind) in self
                .graph
                .edges_directed(output.0, Direction::Outgoing)
                .map(|edge| (edge.target(), *edge.weight()))
                // FIXME: Remove this allocation
                .collect::<Vec<_>>()
            {
                tracing::trace!(
                    "splicing {:?}->{:?} into {:?}->{:?} (kind: {:?}, parent: {:?})",
                    input,
                    output,
                    OutputPort(src),
                    InputPort(dest),
                    src_kind,
                    self.nodes[&self.graph[input.0].parent],
                );
                debug_assert_eq!(src_kind, dest_kind);

                self.graph.add_edge(src, dest, src_kind);
            }
        }

        tracing::trace!("removing old ports {:?}, {:?}", input, output);
        self.graph.remove_node(input.0);
        self.graph.remove_node(output.0);
    }

    #[track_caller]
    fn assert_value_port<P>(&self, port: P)
    where
        P: Port,
    {
        debug_assert!(self.graph.contains_node(port.index()));
        debug_assert_eq!(self.graph[port.index()].edge, EdgeKind::Value);
    }

    #[track_caller]
    fn assert_effect_port<P>(&self, port: P)
    where
        P: Port,
    {
        debug_assert!(self.graph.contains_node(port.index()));
        debug_assert_eq!(self.graph[port.index()].edge, EdgeKind::Effect);
    }

    pub fn start(&mut self) -> Start {
        let start_id = self.next_node();

        let effect = self.output_port(start_id, EdgeKind::Effect);

        let start = Start::new(start_id, effect);
        self.nodes.insert(start_id, Node::Start(start));

        start
    }

    #[track_caller]
    pub fn end(&mut self, effect: OutputPort) -> End {
        self.assert_effect_port(effect);

        let end_id = self.next_node();

        let effect_port = self.input_port(end_id, EdgeKind::Effect);
        self.add_effect_edge(effect, effect_port);

        let end = End::new(end_id, effect_port);
        self.nodes.insert(end_id, Node::End(end));

        end
    }

    pub fn int(&mut self, value: i32) -> Int {
        let int_id = self.next_node();

        let output = self.output_port(int_id, EdgeKind::Value);

        let int = Int::new(int_id, output);
        self.nodes.insert(int_id, Node::Int(int, value));

        int
    }

    pub fn bool(&mut self, value: bool) -> Bool {
        let bool_id = self.next_node();

        let output = self.output_port(bool_id, EdgeKind::Value);

        let bool = Bool::new(bool_id, output);
        self.nodes.insert(bool_id, Node::Bool(bool, value));

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
        self.nodes.insert(add_id, Node::Add(add));

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
        self.nodes.insert(load_id, Node::Load(load));

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
        self.nodes.insert(store_id, Node::Store(store));

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
        self.nodes.insert(input_id, Node::Input(input));

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
        self.nodes.insert(output_id, Node::Output(output));

        output
    }

    fn input_param(&mut self, kind: EdgeKind) -> InputParam {
        let input_id = self.next_node();

        let port = self.output_port(input_id, EdgeKind::Value);
        let param = InputParam::new(input_id, port, kind);
        self.nodes.insert(input_id, Node::InputPort(param));

        param
    }

    fn output_param(&mut self, input: OutputPort, kind: EdgeKind) -> OutputParam {
        let output_id = self.next_node();

        let port = self.input_port(output_id, EdgeKind::Value);
        self.add_edge(input, port, kind);

        let param = OutputParam::new(output_id, port, kind);
        self.nodes.insert(output_id, Node::OutputPort(param));

        param
    }

    #[track_caller]
    pub fn theta<I, F>(&mut self, inputs: I, effect: OutputPort, theta: F) -> Theta
    where
        I: IntoIterator<Item = OutputPort>,
        F: FnOnce(&mut Rvsdg, OutputPort, &[OutputPort]) -> ThetaData,
    {
        self.assert_effect_port(effect);

        let theta_id = self.next_node();

        let effect_in = self.input_port(theta_id, EdgeKind::Effect);
        self.add_effect_edge(effect, effect_in);

        // Wire up the external inputs to the theta node
        let outer_inputs: Vec<_> = inputs
            .into_iter()
            .map(|input| {
                self.assert_value_port(input);

                let port = self.input_port(theta_id, EdgeKind::Value);
                self.add_value_edge(input, port);
                port
            })
            .collect();

        // Create the theta's subgraph
        let mut subgraph = Rvsdg::from_counter(self.counter.clone());

        // Create the input ports within the subgraph
        let (input_params, inner_input_ports): (Vec<_>, Vec<_>) = (0..outer_inputs.len())
            .map(|_| {
                let param = subgraph.input_param(EdgeKind::Value);
                (param.node, param.value())
            })
            .unzip();

        // Create the theta start node
        let start = subgraph.start();
        let ThetaData {
            outputs,
            condition,
            effect,
        } = theta(&mut subgraph, start.effect(), &inner_input_ports);

        subgraph.assert_effect_port(effect);
        let end = subgraph.end(effect);

        subgraph.assert_value_port(condition);
        let condition_param = subgraph.output_param(condition, EdgeKind::Value);

        let output_params: Vec<_> = outputs
            .iter()
            .map(|&output| {
                subgraph.assert_value_port(output);
                subgraph.output_param(output, EdgeKind::Value).node()
            })
            .collect();

        let effect_out = self.output_port(theta_id, EdgeKind::Effect);
        let outer_outputs: Vec<_> = (0..output_params.len())
            .map(|_| self.output_port(theta_id, EdgeKind::Value))
            .collect();

        let theta = Theta::new(
            theta_id,               // node
            outer_inputs,           // inputs
            effect_in,              // effect_in
            input_params,           // input_params
            start.effect(),         // input_effect
            outer_outputs,          // outputs
            effect_out,             // effect_out
            output_params,          // output_params
            start.node,             // start_node
            end.node(),             // end_node
            Box::new(subgraph),     // body
            condition_param.node(), // condition
        );
        self.nodes.insert(theta_id, Node::Theta(theta.clone()));

        theta
    }

    #[track_caller]
    pub fn gamma<I, T, F>(
        &mut self,
        inputs: I,
        effect: OutputPort,
        condition: OutputPort,
        truthy: T,
        falsy: F,
    ) -> Gamma
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
        let outer_inputs: Vec<_> = inputs
            .into_iter()
            .map(|input| {
                self.assert_value_port(input);

                let port = self.input_port(gamma_id, EdgeKind::Value);
                self.add_value_edge(input, port);
                port
            })
            .collect();

        // Create the gamma's true branch
        let mut truthy_subgraph = Rvsdg::from_counter(self.counter.clone());

        // Create the input ports within the subgraph
        let (truthy_input_params, truthy_inner_input_ports): (Vec<_>, Vec<_>) = (0..outer_inputs
            .len())
            .map(|_| {
                let param = truthy_subgraph.input_param(EdgeKind::Value);
                (param.node, param.value())
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

        let truthy_output_params: Vec<_> = truthy_outputs
            .iter()
            .map(|&output| {
                truthy_subgraph.assert_value_port(output);
                truthy_subgraph.output_param(output, EdgeKind::Value).node()
            })
            .collect();

        // Create the gamma's true branch
        let mut falsy_subgraph = Rvsdg::from_counter(self.counter.clone());

        // Create the input ports within the subgraph
        let (falsy_input_params, falsy_inner_input_ports): (Vec<_>, Vec<_>) = (0..outer_inputs
            .len())
            .map(|_| {
                let param = falsy_subgraph.input_param(EdgeKind::Value);
                (param.node, param.value())
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

        let falsy_output_params: Vec<_> = falsy_outputs
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
        let output_params: Vec<_> = truthy_output_params
            .into_iter()
            .zip(falsy_output_params)
            .map(|(truthy, falsy)| [truthy, falsy])
            .collect();

        let effect_out = self.output_port(gamma_id, EdgeKind::Effect);
        let outer_outputs: Vec<_> = (0..output_params.len())
            .map(|_| self.output_port(gamma_id, EdgeKind::Value))
            .collect();

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
        self.nodes.insert(gamma_id, Node::Gamma(gamma.clone()));

        gamma
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
        self.nodes.insert(eq_id, Node::Eq(eq));

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
        self.nodes.insert(not_id, Node::Not(not));

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
        self.nodes.insert(neg_id, Node::Neg(neg));

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

#[derive(Debug, Clone, Copy)]
pub struct EdgeDescriptor {
    effect: EdgeCount,
    value: EdgeCount,
}

impl EdgeDescriptor {
    const fn new(effect: EdgeCount, value: EdgeCount) -> Self {
        Self { effect, value }
    }

    pub const fn effect(&self) -> EdgeCount {
        self.effect
    }

    pub const fn value(&self) -> EdgeCount {
        self.value
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeCount {
    min: Option<usize>,
    max: Option<usize>,
}

impl EdgeCount {
    const fn new(min: Option<usize>, max: Option<usize>) -> Self {
        Self { min, max }
    }

    const fn unlimited() -> Self {
        Self::new(None, None)
    }

    const fn exact(count: usize) -> Self {
        Self::new(Some(count), Some(count))
    }

    const fn zero() -> Self {
        Self::exact(0)
    }

    const fn one() -> Self {
        Self::exact(1)
    }

    const fn two() -> Self {
        Self::exact(2)
    }

    #[allow(dead_code)]
    const fn three() -> Self {
        Self::exact(3)
    }

    pub const fn contains(&self, value: usize) -> bool {
        match (self.min, self.max) {
            (Some(min), None) => value >= min,
            (None, Some(max)) => value <= max,
            (Some(min), Some(max)) => min <= value && value <= max,
            (None, None) => true,
        }
    }

    pub const fn min(&self) -> Option<usize> {
        self.min
    }

    pub const fn max(&self) -> Option<usize> {
        self.max
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
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
    value: OutputPort,
    kind: EdgeKind,
}

impl InputParam {
    const fn new(node: NodeId, value: OutputPort, kind: EdgeKind) -> Self {
        Self { node, value, kind }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutputParam {
    node: NodeId,
    value: InputPort,
    kind: EdgeKind,
}

impl OutputParam {
    const fn new(node: NodeId, value: InputPort, kind: EdgeKind) -> Self {
        Self { node, value, kind }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn value(&self) -> InputPort {
        self.value
    }
}

#[derive(Debug, Clone)]
pub struct Theta {
    node: NodeId,
    inputs: Vec<InputPort>,
    // TODO: Make the input & output effects optional and linked along with the inner effect params
    effect_in: InputPort,
    input_params: Vec<NodeId>,
    input_effect: OutputPort,
    outputs: Vec<OutputPort>,
    effect_out: OutputPort,
    output_params: Vec<NodeId>,
    start_node: NodeId,
    end_node: NodeId,
    body: Box<Rvsdg>,
    condition: NodeId,
}

impl Theta {
    #[allow(clippy::too_many_arguments)]
    const fn new(
        node: NodeId,
        inputs: Vec<InputPort>,
        effect_in: InputPort,
        input_params: Vec<NodeId>,
        input_effect: OutputPort,
        outputs: Vec<OutputPort>,
        effect_out: OutputPort,
        output_params: Vec<NodeId>,
        start_node: NodeId,
        end_node: NodeId,
        body: Box<Rvsdg>,
        condition: NodeId,
    ) -> Self {
        Self {
            node,
            inputs,
            effect_in,
            input_params,
            input_effect,
            outputs,
            effect_out,
            output_params,
            start_node,
            end_node,
            body,
            condition,
        }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub fn inputs(&self) -> &[InputPort] {
        &self.inputs
    }

    pub fn inputs_mut(&mut self) -> &mut Vec<InputPort> {
        &mut self.inputs
    }

    pub fn input_params(&self) -> &[NodeId] {
        &self.input_params
    }

    pub fn input_params_mut(&mut self) -> &mut Vec<NodeId> {
        &mut self.input_params
    }

    pub const fn effect_in(&self) -> InputPort {
        self.effect_in
    }

    pub fn outputs(&self) -> &[OutputPort] {
        &self.outputs
    }

    #[allow(dead_code)]
    pub fn outputs_mut(&mut self) -> &mut Vec<OutputPort> {
        &mut self.outputs
    }

    pub fn output_params(&self) -> &[NodeId] {
        &self.output_params
    }

    #[allow(dead_code)]
    pub fn output_params_mut(&mut self) -> &mut Vec<NodeId> {
        &mut self.output_params
    }

    pub const fn effect_out(&self) -> OutputPort {
        self.effect_out
    }

    pub const fn body(&self) -> &Rvsdg {
        &self.body
    }

    pub fn body_mut(&mut self) -> &mut Rvsdg {
        &mut self.body
    }

    #[allow(dead_code)]
    pub fn into_body(self) -> Rvsdg {
        *self.body
    }

    #[allow(dead_code)]
    pub const fn start(&self) -> NodeId {
        self.start_node
    }

    #[allow(dead_code)]
    pub const fn end(&self) -> NodeId {
        self.end_node
    }

    pub const fn condition(&self) -> NodeId {
        self.condition
    }

    pub fn is_infinite(&self) -> bool {
        let start = self.body().get_node(self.start()).to_start();

        let next_is_end = self
            .body()
            .get_output(start.effect())
            .map_or(false, |(consumer, _, _)| consumer.is_end());

        let condition_output = self.body().get_node(self.condition()).to_output_param();
        let condition_is_false = self
            .body()
            .get_input(condition_output.value())
            .0
            .as_bool()
            .map_or(false, |(_, value)| value);

        next_is_end && condition_is_false
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThetaData {
    outputs: Box<[OutputPort]>,
    condition: OutputPort,
    effect: OutputPort,
}

impl ThetaData {
    pub fn new<O>(outputs: O, condition: OutputPort, effect: OutputPort) -> Self
    where
        O: IntoIterator<Item = OutputPort>,
    {
        Self {
            outputs: outputs.into_iter().collect(),
            condition,
            effect,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Gamma {
    node: NodeId,
    inputs: Vec<InputPort>,
    // TODO: Make optional
    effect_in: InputPort,
    input_params: Vec<[NodeId; 2]>,
    input_effect: OutputPort,
    outputs: Vec<OutputPort>,
    effect_out: OutputPort,
    output_params: Vec<[NodeId; 2]>,
    // TODO: Make optional, linked with `effect_in`
    output_effects: [OutputPort; 2],
    start_nodes: [NodeId; 2],
    end_nodes: [NodeId; 2],
    bodies: Box<[Rvsdg; 2]>,
    condition: InputPort,
}

impl Gamma {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node: NodeId,
        inputs: Vec<InputPort>,
        effect_in: InputPort,
        input_params: Vec<[NodeId; 2]>,
        input_effect: OutputPort,
        outputs: Vec<OutputPort>,
        effect_out: OutputPort,
        output_params: Vec<[NodeId; 2]>,
        output_effects: [OutputPort; 2],
        start_nodes: [NodeId; 2],
        end_nodes: [NodeId; 2],
        bodies: Box<[Rvsdg; 2]>,
        condition: InputPort,
    ) -> Self {
        Self {
            node,
            inputs,
            effect_in,
            input_params,
            input_effect,
            outputs,
            effect_out,
            output_params,
            output_effects,
            start_nodes,
            end_nodes,
            bodies,
            condition,
        }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn starts(&self) -> [NodeId; 2] {
        self.start_nodes
    }

    pub const fn ends(&self) -> [NodeId; 2] {
        self.end_nodes
    }

    pub fn inputs(&self) -> &[InputPort] {
        &self.inputs
    }

    pub const fn condition(&self) -> InputPort {
        self.condition
    }

    pub const fn effect_in(&self) -> InputPort {
        self.effect_in
    }

    pub const fn effect_out(&self) -> OutputPort {
        self.effect_out
    }

    pub fn input_params(&self) -> &[[NodeId; 2]] {
        &self.input_params
    }

    pub fn outputs(&self) -> &[OutputPort] {
        &self.outputs
    }

    #[allow(dead_code)]
    pub fn outputs_mut(&mut self) -> &mut Vec<OutputPort> {
        &mut self.outputs
    }

    pub fn output_params(&self) -> &[[NodeId; 2]] {
        &self.output_params
    }

    pub const fn true_branch(&self) -> &Rvsdg {
        &self.bodies[0]
    }

    pub fn truthy_mut(&mut self) -> &mut Rvsdg {
        &mut self.bodies[0]
    }

    pub const fn false_branch(&self) -> &Rvsdg {
        &self.bodies[1]
    }

    pub fn falsy_mut(&mut self) -> &mut Rvsdg {
        &mut self.bodies[1]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GammaData {
    outputs: Box<[OutputPort]>,
    effect: OutputPort,
}

impl GammaData {
    pub fn new<O>(outputs: O, effect: OutputPort) -> Self
    where
        O: IntoIterator<Item = OutputPort>,
    {
        Self {
            outputs: outputs.into_iter().collect(),
            effect,
        }
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

impl Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NodeId(")?;
        Debug::fmt(&self.0, f)?;
        f.write_char(')')
    }
}

impl Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

pub trait Port: Debug + Clone + Copy + PartialEq + cmp::Eq + Hash {
    fn index(&self) -> NodeIndex;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct InputPort(NodeIndex);

impl Port for InputPort {
    fn index(&self) -> NodeIndex {
        self.0
    }
}

impl Debug for InputPort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("InputPort(")?;
        Debug::fmt(&self.0.index(), f)?;
        f.write_char(')')
    }
}

impl Display for InputPort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0.index(), f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct OutputPort(NodeIndex);

impl Port for OutputPort {
    fn index(&self) -> NodeIndex {
        self.0
    }
}

impl Debug for OutputPort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("OutputPort(")?;
        Debug::fmt(&self.0.index(), f)?;
        f.write_char(')')
    }
}

impl Display for OutputPort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0.index(), f)
    }
}
