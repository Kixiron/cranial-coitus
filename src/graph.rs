use petgraph::{
    dot::Dot,
    stable_graph::{NodeIndex, StableGraph},
    visit::EdgeRef,
    Direction,
};
use std::{
    cmp,
    collections::HashMap,
    fmt::{self, Debug, Display, Write as _},
    fs::{self, File},
    hash::Hash,
    path::Path,
};

#[derive(Debug, Clone)]
pub struct Rvsdg {
    graph: StableGraph<PortData, EdgeKind>,
    nodes: HashMap<NodeId, Node>,
    node_counter: u32,
}

impl Rvsdg {
    pub fn new() -> Self {
        Self {
            graph: StableGraph::new(),
            nodes: HashMap::new(),
            node_counter: 0,
        }
    }

    pub fn to_dot(&self, path: impl AsRef<Path>) {
        use std::io::Write;

        let path = path.as_ref();
        let dot = Dot::new(&self.graph);

        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut file = File::create(path).unwrap();

        write!(&mut file, "{}", dot).unwrap();
    }

    fn next_node(&mut self) -> NodeId {
        let node = NodeId(self.node_counter);
        self.node_counter += 1;

        node
    }

    fn add_edge(&mut self, src: OutputPort, dest: InputPort, kind: EdgeKind) {
        // TODO: Add some more invariant assertions here, ideally for
        //       ensuring that we're creating the expected edge type for
        //       the given ports
        debug_assert_eq!(self.graph[src.0].kind, PortKind::Output);
        debug_assert_eq!(self.graph[dest.0].kind, PortKind::Input);

        self.graph.add_edge(src.0, dest.0, kind);
    }

    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn contains_node(&self, node: NodeId) -> bool {
        self.nodes.contains_key(&node)
    }

    pub fn add_value_edge(&mut self, src: OutputPort, dest: InputPort) {
        self.add_edge(src, dest, EdgeKind::Value);
    }

    pub fn add_effect_edge(&mut self, src: OutputPort, dest: InputPort) {
        self.add_edge(src, dest, EdgeKind::Effect);
    }

    fn input_port(&mut self, parent: NodeId) -> InputPort {
        InputPort(self.graph.add_node(PortData::input(parent)))
    }

    fn output_port(&mut self, parent: NodeId) -> OutputPort {
        OutputPort(self.graph.add_node(PortData::output(parent)))
    }

    pub fn nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.keys().copied()
    }

    pub fn try_node(&self, node: NodeId) -> Option<&Node> {
        self.nodes.get(&node)
    }

    #[track_caller]
    pub fn get_node(&self, node: NodeId) -> &Node {
        self.try_node(node)
            .expect("tried to get node that doesn't exist")
    }

    pub fn try_node_mut(&mut self, node: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&node)
    }

    pub fn get_node_mut(&mut self, node: NodeId) -> &mut Node {
        self.try_node_mut(node).unwrap()
    }

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

    pub fn input_source(&self, input: InputPort) -> OutputPort {
        let mut incoming = self.graph.edges_directed(input.0, Direction::Incoming);
        debug_assert_eq!(incoming.clone().count(), 1);

        OutputPort(incoming.next().unwrap().source())
    }

    pub fn get_input(&self, input: InputPort) -> (&Node, OutputPort, EdgeKind) {
        self.try_input(input).unwrap_or_else(|| {
            let port = self.graph[input.0];

            panic!(
                "incorrect number of edges found for input port {:?} (port data: {:?}, node: {:?})",
                input, port, self.nodes[&port.parent],
            );
        })
    }

    pub fn try_input(&self, input: InputPort) -> Option<(&Node, OutputPort, EdgeKind)> {
        let port = self.graph[input.0];
        debug_assert_eq!(port.kind, PortKind::Input);

        let mut incoming = self.graph.edges_directed(input.0, Direction::Incoming);
        // FIXME: debug_assert_matches!()
        debug_assert!(
            matches!(incoming.clone().count(), 0 | 1),
            "incorrect number of edges found for input port {:?} (port data: {:?}, node: {:?})",
            input,
            port,
            self.nodes[&port.parent],
        );

        incoming.next().map(|edge| {
            let (src, kind) = (edge.source(), *edge.weight());
            (&self.nodes[&self.graph[src].parent], OutputPort(src), kind)
        })
    }

    pub fn get_output(&self, output: OutputPort) -> Option<(&Node, EdgeKind)> {
        let port = self.graph[output.0];
        debug_assert_eq!(port.kind, PortKind::Output);

        self.graph
            .edges_directed(output.0, Direction::Outgoing)
            .next()
            .map(|edge| {
                let dest = edge.target();
                let node = &self.nodes[&self.graph[dest].parent];

                (node, *edge.weight())
            })
    }

    pub fn inputs(
        &self,
        node: NodeId,
    ) -> impl Iterator<Item = (InputPort, &Node, OutputPort, EdgeKind)> {
        let node = self.get_node(node);

        node.inputs()
            .into_iter()
            .map(move |input| {
                let port = self.graph[input.0];
                let (node, output, kind) = self.try_input(input)
                    .unwrap_or_else(|| panic!(
                        "incorrect number of edges found for input port {:?} (port data: {:?}, node: {:?})",
                        input,
                        port,
                        self.nodes[&port.parent],
                    ));

                (input, node, output, kind)
            })
    }

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
    ) -> impl Iterator<Item = (OutputPort, Option<(&Node, EdgeKind)>)> {
        self.get_node(node)
            .outputs()
            .into_iter()
            .map(|output| (output, self.get_output(output)))
    }

    pub fn remove_input(&mut self, input: InputPort) {
        let mut incoming = self.graph.edges_directed(input.0, Direction::Incoming);
        debug_assert_eq!(incoming.clone().count(), 1);

        let edge = incoming.next().unwrap().id();
        self.graph.remove_edge(edge);
    }

    pub fn remove_output_edge(&mut self, output: OutputPort) {
        let outgoing = self
            .graph
            .edges_directed(output.0, Direction::Outgoing)
            .next()
            .map(|edge| edge.id());

        if let Some(edge) = outgoing {
            println!("removing output port's edge {} (edge: {:?})", output, edge);
            self.graph.remove_edge(edge);
        } else {
            println!("output {} doesn't exist, not removing edge", output);
        }
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

    pub fn rewire_dependents(&mut self, old_port: OutputPort, rewire_to: OutputPort) {
        let edges: Vec<_> = self
            .graph
            .edges_directed(old_port.0, Direction::Outgoing)
            .map(|edge| (edge.id(), edge.target(), *edge.weight()))
            .collect();

        for (edge, dest, kind) in edges {
            self.graph.remove_edge(edge);
            self.graph.add_edge(rewire_to.0, dest, kind);

            // let parent = self.graph[dest].parent;
            // for input in self.nodes.get_mut(&parent).unwrap().inputs_mut() {
            //     if *input == InputPort(dest) {
            //         println!("replaced {:?} on {:?} to point to {:?}")
            //         *input = InputPort(dest);
            //     }
            // }
        } //
    }

    pub fn splice_ports(&mut self, input: InputPort, output: OutputPort) {
        debug_assert_eq!(self.port_parent(input), self.port_parent(output));

        let mut incoming = self.graph.edges_directed(input.0, Direction::Incoming);
        // FIXME: `debug_assert_matches!()`
        debug_assert!(matches!(incoming.clone().count(), 1 | 0));

        if let Some((src, src_kind)) = incoming.next().map(|edge| (edge.source(), *edge.weight())) {
            for (dest, dest_kind) in self
                .graph
                .edges_directed(output.0, Direction::Outgoing)
                .map(|edge| (edge.target(), *edge.weight()))
                .collect::<Vec<_>>()
            {
                println!(
                    "splicing {:?}->{:?} into {:?}->{:?} (kind: {:?}, parent: {:?}, inputs: {:?}, outputs: {:?})",
                    input,
                    output,
                    OutputPort(src),
                    InputPort(dest),
                    src_kind,
                    self.nodes[&self.graph[input.0].parent],
                    self.inputs(self.graph[input.0].parent)
                        .map(|(port, _, _, _)| port)
                        .collect::<Vec<_>>(),
                    self.outputs(self.graph[input.0].parent)
                        .map(|(port, _)| port)
                        .collect::<Vec<_>>(),
                );
                debug_assert_eq!(src_kind, dest_kind);

                self.graph.add_edge(src, dest, src_kind);
            }
        }

        println!("removing old ports {:?}, {:?}", input, output);
        self.graph.remove_node(input.0);
        self.graph.remove_node(output.0);
    }

    pub fn start(&mut self) -> Start {
        let start_id = self.next_node();

        let effect = self.output_port(start_id);

        let start = Start::new(start_id, effect);
        self.nodes.insert(start_id, Node::Start(start));

        start
    }

    pub fn end(&mut self, effect: OutputPort) -> End {
        let end_id = self.next_node();

        let effect_port = self.input_port(end_id);
        self.add_effect_edge(effect, effect_port);

        let end = End::new(end_id, effect_port);
        self.nodes.insert(end_id, Node::End(end));

        end
    }

    pub fn int(&mut self, value: i32) -> Int {
        let int_id = self.next_node();

        let output = self.output_port(int_id);

        let int = Int::new(int_id, output);
        self.nodes.insert(int_id, Node::Int(int, value));

        int
    }

    pub fn bool(&mut self, value: bool) -> Bool {
        let bool_id = self.next_node();

        let output = self.output_port(bool_id);

        let bool = Bool::new(bool_id, output);
        self.nodes.insert(bool_id, Node::Bool(bool, value));

        bool
    }

    pub fn add(&mut self, lhs: OutputPort, rhs: OutputPort) -> Add {
        let add_id = self.next_node();

        let lhs_port = self.input_port(add_id);
        self.add_value_edge(lhs, lhs_port);

        let rhs_port = self.input_port(add_id);
        self.add_value_edge(rhs, rhs_port);

        let output = self.output_port(add_id);

        let add = Add::new(add_id, lhs_port, rhs_port, output);
        self.nodes.insert(add_id, Node::Add(add));

        add
    }

    pub fn load(&mut self, ptr: OutputPort, effect: OutputPort) -> Load {
        let load_id = self.next_node();

        let ptr_port = self.input_port(load_id);
        self.add_value_edge(ptr, ptr_port);

        let effect_port = self.input_port(load_id);
        self.add_effect_edge(effect, effect_port);

        let loaded = self.output_port(load_id);
        let effect_out = self.output_port(load_id);

        let load = Load::new(load_id, ptr_port, effect_port, loaded, effect_out);
        self.nodes.insert(load_id, Node::Load(load));

        load
    }

    pub fn store(&mut self, ptr: OutputPort, value: OutputPort, effect: OutputPort) -> Store {
        let store_id = self.next_node();

        let ptr_port = self.input_port(store_id);
        self.add_value_edge(ptr, ptr_port);

        let value_port = self.input_port(store_id);
        self.add_value_edge(value, value_port);

        let effect_port = self.input_port(store_id);
        self.add_effect_edge(effect, effect_port);

        let effect_out = self.output_port(store_id);

        let store = Store::new(store_id, ptr_port, value_port, effect_port, effect_out);
        self.nodes.insert(store_id, Node::Store(store));

        store
    }

    pub fn input(&mut self, effect: OutputPort) -> Input {
        let input_id = self.next_node();

        let effect_port = self.input_port(input_id);
        self.add_effect_edge(effect, effect_port);

        let value = self.output_port(input_id);
        let effect_out = self.output_port(input_id);

        let input = Input::new(input_id, effect_port, value, effect_out);
        self.nodes.insert(input_id, Node::Input(input));

        input
    }

    pub fn output(&mut self, value: OutputPort, effect: OutputPort) -> Output {
        let output_id = self.next_node();

        let value_port = self.input_port(output_id);
        self.add_value_edge(value, value_port);

        let effect_port = self.input_port(output_id);
        self.add_effect_edge(effect, effect_port);

        let effect_out = self.output_port(output_id);

        let output = Output::new(output_id, value_port, effect_port, effect_out);
        self.nodes.insert(output_id, Node::Output(output));

        output
    }

    fn input_param(&mut self, kind: EdgeKind) -> InputParam {
        let input_id = self.next_node();

        let port = self.output_port(input_id);
        let param = InputParam::new(input_id, port, kind);
        self.nodes.insert(input_id, Node::InputPort(param));

        param
    }

    fn output_param(&mut self, input: OutputPort, kind: EdgeKind) -> OutputParam {
        let output_id = self.next_node();

        let port = self.input_port(output_id);
        self.add_edge(input, port, kind);

        let param = OutputParam::new(output_id, port, kind);
        self.nodes.insert(output_id, Node::OutputPort(param));

        param
    }

    pub fn theta<I, F>(&mut self, inputs: I, effect: OutputPort, theta: F) -> Theta
    where
        I: IntoIterator<Item = OutputPort>,
        F: FnOnce(&mut Rvsdg, OutputPort, &[OutputPort]) -> ThetaData,
    {
        let theta_id = self.next_node();

        let effect_in = self.input_port(theta_id);
        self.add_effect_edge(effect, effect_in);

        // Wire up the external inputs to the theta node
        let outer_inputs: Vec<_> = inputs
            .into_iter()
            .map(|input| {
                let port = self.input_port(theta_id);
                self.add_value_edge(input, port);
                port
            })
            .collect();

        // Create the theta's subgraph
        let mut subgraph = Rvsdg::new();
        subgraph.node_counter = self.node_counter;

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
            effect: output_effect,
        } = theta(&mut subgraph, start.effect(), &inner_input_ports);

        let _end = subgraph.end(output_effect);

        let output_params: Vec<_> = outputs
            .iter()
            .map(|&output| subgraph.output_param(output, EdgeKind::Value).node)
            .collect();

        self.node_counter = subgraph.node_counter;

        let effect_out = self.output_port(theta_id);
        let outer_outputs: Vec<_> = (0..output_params.len())
            .map(|_| self.output_port(theta_id))
            .collect();

        let theta = Theta::new(
            theta_id,           // node
            outer_inputs,       // inputs
            effect_in,          // effect_in
            input_params,       // input_params
            start.effect(),     // input_effect
            outer_outputs,      // outputs
            effect_out,         // effect_out
            output_params,      // output_params
            output_effect,      // output_effect
            start.node,         // start_node
            Box::new(subgraph), // body
            condition,          // condition
        );
        self.nodes.insert(theta_id, Node::Theta(theta.clone()));

        theta
    }

    pub fn phi<I, T, F>(
        &mut self,
        inputs: I,
        effect: OutputPort,
        condition: OutputPort,
        truthy: T,
        falsy: F,
    ) -> Phi
    where
        I: IntoIterator<Item = OutputPort>,
        T: FnOnce(&mut Rvsdg, OutputPort, &[OutputPort]) -> PhiData,
        F: FnOnce(&mut Rvsdg, OutputPort, &[OutputPort]) -> PhiData,
    {
        let phi_id = self.next_node();

        let effect_in = self.input_port(phi_id);
        self.add_effect_edge(effect, effect_in);

        let cond_port = self.input_port(phi_id);
        self.add_value_edge(condition, cond_port);

        // Wire up the external inputs to the phi node
        let outer_inputs: Vec<_> = inputs
            .into_iter()
            .map(|input| {
                let port = self.input_port(phi_id);
                self.add_value_edge(input, port);
                port
            })
            .collect();

        // Create the phi's true branch
        let mut truthy_subgraph = Rvsdg::new();
        truthy_subgraph.node_counter = self.node_counter;

        // Create the input ports within the subgraph
        let (truthy_input_params, truthy_inner_input_ports): (Vec<_>, Vec<_>) = (0..outer_inputs
            .len())
            .map(|_| {
                let param = truthy_subgraph.input_param(EdgeKind::Value);
                (param.node, param.value())
            })
            .unzip();

        // Create the branch's start node
        let start = truthy_subgraph.start();
        let PhiData {
            outputs: truthy_outputs,
            effect: truthy_output_effect,
        } = truthy(
            &mut truthy_subgraph,
            start.effect(),
            &truthy_inner_input_ports,
        );

        let _end = truthy_subgraph.end(truthy_output_effect);

        let truthy_output_params: Vec<_> = truthy_outputs
            .iter()
            .map(|&output| truthy_subgraph.output_param(output, EdgeKind::Value).node)
            .collect();

        self.node_counter = truthy_subgraph.node_counter;

        // Create the phi's true branch
        let mut falsy_subgraph = Rvsdg::new();
        falsy_subgraph.node_counter = self.node_counter;

        // Create the input ports within the subgraph
        let (falsy_input_params, falsy_inner_input_ports): (Vec<_>, Vec<_>) = (0..outer_inputs
            .len())
            .map(|_| {
                let param = falsy_subgraph.input_param(EdgeKind::Value);
                (param.node, param.value())
            })
            .unzip();

        // Create the branch's start node
        let start = falsy_subgraph.start();
        let PhiData {
            outputs: falsy_outputs,
            effect: falsy_output_effect,
        } = falsy(
            &mut falsy_subgraph,
            start.effect(),
            &falsy_inner_input_ports,
        );

        let _end = falsy_subgraph.end(falsy_output_effect);

        let falsy_output_params: Vec<_> = falsy_outputs
            .iter()
            .map(|&output| falsy_subgraph.output_param(output, EdgeKind::Value).node)
            .collect();

        self.node_counter = falsy_subgraph.node_counter;

        // FIXME: I'd really like to be able to support variable numbers of inputs for each branch
        //        to allow some more flexible optimizations like removing effect flow from a branch
        debug_assert_eq!(truthy_input_params.len(), falsy_input_params.len());
        let input_params = truthy_input_params
            .into_iter()
            .zip(falsy_input_params)
            .map(|(truthy, falsy)| [truthy, falsy])
            .collect();

        debug_assert_eq!(truthy_output_params.len(), falsy_output_params.len());
        let output_params: Vec<_> = truthy_output_params
            .into_iter()
            .zip(falsy_output_params)
            .map(|(truthy, falsy)| [truthy, falsy])
            .collect();

        let effect_out = self.output_port(phi_id);
        let outer_outputs: Vec<_> = (0..output_params.len())
            .map(|_| self.output_port(phi_id))
            .collect();

        let phi = Phi::new(
            phi_id,                                      // node
            outer_inputs,                                // inputs
            effect_in,                                   // effect_in
            input_params,                                // input_params
            start.effect(),                              // input_effect
            outer_outputs,                               // outputs
            effect_out,                                  // effect_out
            output_params,                               // output_params
            [truthy_output_effect, falsy_output_effect], // output_effect
            start.node,                                  // start_node
            Box::new([truthy_subgraph, falsy_subgraph]), // body
            cond_port,                                   // condition
        );
        self.nodes.insert(phi_id, Node::Phi(phi.clone()));

        phi
    }

    pub fn eq(&mut self, lhs: OutputPort, rhs: OutputPort) -> Eq {
        let eq_id = self.next_node();

        let lhs_port = self.input_port(eq_id);
        self.add_value_edge(lhs, lhs_port);

        let rhs_port = self.input_port(eq_id);
        self.add_value_edge(rhs, rhs_port);

        let output = self.output_port(eq_id);

        let eq = Eq::new(eq_id, lhs_port, rhs_port, output);
        self.nodes.insert(eq_id, Node::Eq(eq));

        eq
    }

    pub fn not(&mut self, input: OutputPort) -> Not {
        let not_id = self.next_node();

        let input_port = self.input_port(not_id);
        self.add_value_edge(input, input_port);

        let output = self.output_port(not_id);

        let not = Not::new(not_id, input_port, output);
        self.nodes.insert(not_id, Node::Not(not));

        not
    }

    pub fn neq(&mut self, lhs: OutputPort, rhs: OutputPort) -> Not {
        let eq = self.eq(lhs, rhs);
        self.not(eq.value())
    }
}

#[derive(Debug, Clone)]
pub enum Node {
    Int(Int, i32),
    Bool(Bool, bool),
    Array(Array, u16),
    Add(Add),
    Load(Load),
    Store(Store),
    Start(Start),
    End(End),
    Input(Input),
    Output(Output),
    Theta(Theta),
    InputPort(InputParam),
    OutputPort(OutputParam),
    Eq(Eq),
    Not(Not),
    Phi(Phi),
}

impl From<InputParam> for Node {
    fn from(v: InputParam) -> Self {
        Self::InputPort(v)
    }
}

impl From<Theta> for Node {
    fn from(v: Theta) -> Self {
        Self::Theta(v)
    }
}

impl From<Output> for Node {
    fn from(v: Output) -> Self {
        Self::Output(v)
    }
}

impl From<Input> for Node {
    fn from(v: Input) -> Self {
        Self::Input(v)
    }
}

impl From<End> for Node {
    fn from(v: End) -> Self {
        Self::End(v)
    }
}

impl From<Start> for Node {
    fn from(v: Start) -> Self {
        Self::Start(v)
    }
}

impl From<Store> for Node {
    fn from(v: Store) -> Self {
        Self::Store(v)
    }
}

impl From<Load> for Node {
    fn from(v: Load) -> Self {
        Self::Load(v)
    }
}

impl From<Add> for Node {
    fn from(v: Add) -> Self {
        Self::Add(v)
    }
}

impl From<OutputParam> for Node {
    fn from(v: OutputParam) -> Self {
        Self::OutputPort(v)
    }
}

impl From<Eq> for Node {
    fn from(v: Eq) -> Self {
        Self::Eq(v)
    }
}

impl From<Not> for Node {
    fn from(v: Not) -> Self {
        Self::Not(v)
    }
}

impl From<Phi> for Node {
    fn from(v: Phi) -> Self {
        Self::Phi(v)
    }
}

impl Node {
    pub fn node_id(&self) -> NodeId {
        match *self {
            Self::Int(Int { node, .. }, _)
            | Self::Bool(Bool { node, .. }, _)
            | Self::Array(Array { node, .. }, _)
            | Self::Add(Add { node, .. })
            | Self::Load(Load { node, .. })
            | Self::Store(Store { node, .. })
            | Self::Start(Start { node, .. })
            | Self::End(End { node, .. })
            | Self::Input(Input { node, .. })
            | Self::Output(Output { node, .. })
            | Self::Theta(Theta { node, .. })
            | Self::InputPort(InputParam { node, .. })
            | Self::OutputPort(OutputParam { node, .. })
            | Self::Eq(Eq { node, .. })
            | Self::Not(Not { node, .. })
            | Self::Phi(Phi { node, .. }) => node,
        }
    }

    // FIXME: TinyVec?
    pub fn inputs(&self) -> Vec<InputPort> {
        match self {
            Self::Int(_, _) | Self::Bool(_, _) => Vec::new(),
            Self::Array(array, _) => array.elements.to_vec(),
            Self::Add(add) => vec![add.lhs, add.rhs],
            Self::Load(load) => vec![load.ptr, load.effect_in],
            Self::Store(store) => vec![store.ptr, store.value, store.effect_in],
            Self::Start(_) => Vec::new(),
            Self::End(end) => vec![end.effect],
            Self::Input(input) => vec![input.effect_in],
            Self::Output(output) => vec![output.value, output.effect_in],
            Self::Theta(theta) => {
                let mut inputs = theta.inputs.to_vec();
                inputs.push(theta.effect_in);
                inputs
            }
            Self::InputPort(_) => Vec::new(),
            Self::OutputPort(output) => vec![output.value],
            Self::Eq(eq) => vec![eq.lhs, eq.rhs],
            Self::Not(not) => vec![not.input],
            Self::Phi(phi) => {
                let mut inputs = phi.inputs.to_vec();
                inputs.push(phi.condition);
                inputs.push(phi.effect_in);
                inputs
            }
        }
    }

    // FIXME: TinyVec?
    pub fn inputs_mut(&mut self) -> Vec<&mut InputPort> {
        match self {
            Self::Int(_, _) | Self::Bool(_, _) => Vec::new(),
            Self::Array(array, _) => array.elements.iter_mut().collect(),
            Self::Add(add) => vec![&mut add.lhs, &mut add.rhs],
            Self::Load(load) => vec![&mut load.ptr, &mut load.effect_in],
            Self::Store(store) => vec![&mut store.ptr, &mut store.value, &mut store.effect_in],
            Self::Start(_) => Vec::new(),
            Self::End(end) => vec![&mut end.effect],
            Self::Input(input) => vec![&mut input.effect_in],
            Self::Output(output) => vec![&mut output.value, &mut output.effect_in],
            Self::Theta(theta) => {
                let mut inputs: Vec<_> = theta.inputs.iter_mut().collect();
                inputs.push(&mut theta.effect_in);
                inputs
            }
            Self::InputPort(_) => Vec::new(),
            Self::OutputPort(output) => vec![&mut output.value],
            Self::Eq(eq) => vec![&mut eq.lhs, &mut eq.rhs],
            Self::Not(not) => vec![&mut not.input],
            Self::Phi(phi) => {
                let mut inputs: Vec<_> = phi.inputs.iter_mut().collect();
                inputs.push(&mut phi.condition);
                inputs.push(&mut phi.effect_in);
                inputs
            }
        }
    }

    // FIXME: TinyVec?
    pub fn outputs(&self) -> Vec<OutputPort> {
        match self {
            Self::Int(int, _) => vec![int.value],
            Self::Bool(bool, _) => vec![bool.value],
            Self::Array(array, _) => vec![array.value],
            Self::Add(add) => vec![add.value],
            Self::Load(load) => vec![load.value, load.effect_out],
            Self::Store(store) => vec![store.effect_out],
            Self::Start(start) => vec![start.effect],
            Self::End(_) => Vec::new(),
            Self::Input(input) => vec![input.value, input.effect_out],
            Self::Output(output) => vec![output.effect_out],
            Self::Theta(theta) => {
                let mut inputs = theta.outputs.to_vec();
                inputs.push(theta.effect_out);
                inputs
            }
            Self::InputPort(input) => vec![input.value],
            Self::OutputPort(_) => Vec::new(),
            Self::Eq(eq) => vec![eq.value],
            Self::Not(not) => vec![not.value],
            Self::Phi(phi) => {
                let mut inputs = phi.outputs.to_vec();
                inputs.push(phi.effect_out);
                inputs
            }
        }
    }

    pub fn input_desc(&self) -> EdgeDescriptor {
        match self {
            Self::Int(..) | Self::Bool(..) | Self::Start(_) | Self::InputPort(_) => {
                EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::zero())
            }
            Self::Add(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::two()),
            Self::Load(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::one()),
            Self::Store(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::two()),
            &Self::Array(_, len) => {
                EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::exact(len as usize))
            }
            Self::End(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
            Self::Input(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::one()),
            Self::Output(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
            Self::Theta(theta) => {
                EdgeDescriptor::new(EdgeCount::one(), EdgeCount::exact(theta.inputs().len()))
            }
            Self::OutputPort(output) => match output.kind {
                EdgeKind::Effect => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
                EdgeKind::Value => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
            },
            Self::Eq(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::two()),
            Self::Not(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
            Self::Phi(phi) => {
                EdgeDescriptor::new(EdgeCount::one(), EdgeCount::exact(phi.inputs().len() + 1))
            }
        }
    }

    pub fn output_desc(&self) -> EdgeDescriptor {
        match self {
            Self::Int(..)
            | Self::Bool(..)
            | Self::Add(_)
            | Self::Array(..)
            | Self::OutputPort(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::unlimited()),
            Self::Load(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::unlimited()),
            Self::Store(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
            Self::Start(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
            Self::End(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::zero()),
            Self::Input(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
            Self::Output(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::one()),
            Self::Theta(theta) => EdgeDescriptor::new(
                EdgeCount::one(),
                EdgeCount::new(None, Some(theta.outputs().len())),
            ),
            Self::InputPort(output) => match output.kind {
                EdgeKind::Effect => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
                EdgeKind::Value => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
            },
            Self::Eq(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
            Self::Not(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
            Self::Phi(phi) => {
                EdgeDescriptor::new(EdgeCount::one(), EdgeCount::exact(phi.outputs().len()))
            }
        }
    }

    /// Returns `true` if the node is [`Int`].
    ///
    /// [`Int`]: Node::Int
    pub const fn is_int(&self) -> bool {
        matches!(self, Self::Int(..))
    }

    pub const fn as_int(&self) -> Option<(Int, i32)> {
        if let Self::Int(int, val) = *self {
            Some((int, val))
        } else {
            None
        }
    }

    pub const fn as_bool(&self) -> Option<(Bool, bool)> {
        if let Self::Bool(bool, val) = *self {
            Some((bool, val))
        } else {
            None
        }
    }

    pub const fn as_store(&self) -> Option<Store> {
        if let Self::Store(store) = *self {
            Some(store)
        } else {
            None
        }
    }

    /// Returns `true` if the node is [`Store`].
    ///
    /// [`Store`]: Node::Store
    pub const fn is_store(&self) -> bool {
        matches!(self, Self::Store(..))
    }

    /// Returns `true` if the node is [`End`].
    ///
    /// [`End`]: Node::End
    pub const fn is_end(&self) -> bool {
        matches!(self, Self::End(..))
    }

    /// Returns `true` if the node is [`Start`].
    ///
    /// [`Start`]: Node::Start
    pub const fn is_start(&self) -> bool {
        matches!(self, Self::Start(..))
    }

    #[track_caller]
    pub fn to_theta_mut(&mut self) -> &mut Theta {
        if let Self::Theta(theta) = self {
            theta
        } else {
            panic!("attempted to get theta, got {:?}", self);
        }
    }

    #[track_caller]
    pub fn to_phi_mut(&mut self) -> &mut Phi {
        if let Self::Phi(phi) = self {
            phi
        } else {
            panic!("attempted to get phi, got {:?}", self);
        }
    }

    #[track_caller]
    pub fn to_store_mut(&mut self) -> &mut Store {
        if let Self::Store(store) = self {
            store
        } else {
            panic!("attempted to get store, got {:?}", self);
        }
    }

    /// Returns `true` if the node is [`OutputPort`].
    ///
    /// [`OutputPort`]: Node::OutputPort
    pub const fn is_output_port(&self) -> bool {
        matches!(self, Self::OutputPort(..))
    }

    #[track_caller]
    pub fn to_int(&self) -> Int {
        if let Self::Int(int, _) = *self {
            int
        } else {
            panic!("attempted to get int, got {:?}", self);
        }
    }

    #[track_caller]
    pub fn to_bool(&self) -> Bool {
        if let Self::Bool(bool, _) = *self {
            bool
        } else {
            panic!("attempted to get bool, got {:?}", self);
        }
    }

    #[track_caller]
    pub fn to_input_param(&self) -> InputParam {
        if let Self::InputPort(param) = *self {
            param
        } else {
            panic!("attempted to get input port, got {:?}", self);
        }
    }

    #[track_caller]
    pub fn to_output_param(&self) -> OutputParam {
        if let Self::OutputPort(param) = *self {
            param
        } else {
            panic!("attempted to get output port, got {:?}", self);
        }
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

    const fn three() -> Self {
        Self::exact(3)
    }

    pub const fn contains(&self, value: usize) -> bool {
        match (self.min, self.max) {
            (Some(min), None) => value >= min,
            (None, Some(max)) => value <= max,
            (Some(min), Some(max)) => value >= min && value <= max,
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
    parent: NodeId,
}

impl PortData {
    const fn new(kind: PortKind, node: NodeId) -> Self {
        Self { kind, parent: node }
    }

    const fn input(node: NodeId) -> Self {
        Self::new(PortKind::Input, node)
    }

    const fn output(node: NodeId) -> Self {
        Self::new(PortKind::Output, node)
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

    /// Get the add's right hand side
    pub const fn rhs(&self) -> InputPort {
        self.rhs
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

#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    node: NodeId,
    elements: Box<[InputPort]>,
    value: OutputPort,
}

impl Array {
    const fn new(node: NodeId, elements: Box<[InputPort]>, value: OutputPort) -> Self {
        Self {
            node,
            elements,
            value,
        }
    }

    pub const fn elements(&self) -> &[InputPort] {
        &self.elements
    }

    pub const fn output(&self) -> OutputPort {
        self.value
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
    output_effect: OutputPort,
    start_node: NodeId,
    body: Box<Rvsdg>,
    condition: OutputPort,
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
        output_effect: OutputPort,
        start_node: NodeId,
        body: Box<Rvsdg>,
        condition: OutputPort,
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
            output_effect,
            start_node,
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

    pub fn input_params(&self) -> &[NodeId] {
        &self.input_params
    }

    pub const fn effect_in(&self) -> InputPort {
        self.effect_in
    }

    pub fn outputs(&self) -> &[OutputPort] {
        &self.outputs
    }

    pub fn outputs_mut(&mut self) -> &mut Vec<OutputPort> {
        &mut self.outputs
    }

    pub fn output_params(&self) -> &[NodeId] {
        &self.output_params
    }

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

    pub fn into_body(self) -> Rvsdg {
        *self.body
    }

    pub const fn start_node(&self) -> NodeId {
        self.start_node
    }

    pub const fn condition(&self) -> OutputPort {
        self.condition
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
pub struct Phi {
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
    start_node: NodeId,
    bodies: Box<[Rvsdg; 2]>,
    condition: InputPort,
}

impl Phi {
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
        start_node: NodeId,
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
            start_node,
            bodies,
            condition,
        }
    }

    pub const fn node(&self) -> NodeId {
        self.node
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

    pub fn outputs_mut(&mut self) -> &mut Vec<OutputPort> {
        &mut self.outputs
    }

    pub fn output_params(&self) -> &[[NodeId; 2]] {
        &self.output_params
    }

    pub const fn truthy(&self) -> &Rvsdg {
        &self.bodies[0]
    }

    pub fn truthy_mut(&mut self) -> &mut Rvsdg {
        &mut self.bodies[0]
    }

    pub const fn falsy(&self) -> &Rvsdg {
        &self.bodies[1]
    }

    pub fn falsy_mut(&mut self) -> &mut Rvsdg {
        &mut self.bodies[1]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhiData {
    outputs: Box<[OutputPort]>,
    effect: OutputPort,
}

impl PhiData {
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
