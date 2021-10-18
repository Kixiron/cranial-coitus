mod associative_add;
mod const_dedup;
mod const_folding;
mod const_loads;
mod dce;
mod eliminate_const_phi;
mod unobserved_store;

pub use associative_add::AssociativeAdd;
pub use const_dedup::ConstDedup;
pub use const_folding::ConstFolding;
pub use const_loads::ConstLoads;
pub use dce::Dce;
pub use eliminate_const_phi::ElimConstPhi;
pub use unobserved_store::UnobservedStore;

use crate::graph::{
    Add, Bool, End, Eq, Input, InputParam, Int, Load, Node, NodeId, Not, Output, OutputParam, Phi,
    Rvsdg, Start, Store, Theta,
};
use std::collections::{HashSet, VecDeque};

// TODO:
// - Technically speaking, if the program contains no `input()` or `output()` invocations
//   and does not infinitely loop it has no *observable* side effects and we can thusly
//   delete it all...
// - Redundant loads propagating through other effect kinds
//   ```
//   _578 := load _577
//   call output(_578)
//   _579 := load _577
//   ```
// - Loop unrolling: We can always extract the first iteration of a loop into something like this
//   ```
//   // *ptr is known not to be zero
//   <body>
//   if *ptr != 0 {
//     do { <body> } while *ptr != 0
//   }
//   ```
//   provided that we know the condition (`*ptr` in the above) is non-zero. This should
//   theoretically allow us to slowly unroll and fully (or at very least partially)
//   evaluate loops in more generalized situations
// - Data flow propagation: If we know the input
// - Removing equivalent nodes (CSE)
// - Region ingress/egress elimination (remove unused input and output edges
//   that go into regions, including effect edges (but only when it's unused in
//   all sub-regions))
pub trait Pass {
    /// The name of the current pass
    fn pass_name(&self) -> &str;

    fn did_change(&self) -> bool;

    fn reset(&mut self);

    fn visit_graph(&mut self, graph: &mut Rvsdg) -> bool {
        self.visit_graph_inner(graph, Vec::new(), &mut VecDeque::new())
    }

    fn visit_graph_inner(
        &mut self,
        graph: &mut Rvsdg,
        mut stack_init: Vec<NodeId>,
        stack: &mut VecDeque<NodeId>,
    ) -> bool {
        stack_init.sort_unstable();

        let (mut visited, mut buffer) = (HashSet::with_capacity(graph.total_nodes()), Vec::new());

        // Initialize the stack with all of the start nodes within the graph
        stack.extend(stack_init);
        stack.extend(graph.nodes().filter_map(|node_id| {
            let node = graph.get_node(node_id);

            (node.is_start() || node.is_end() || node.is_input_port() || node.is_output_port())
                .then(|| node_id)
        }));
        // Sort for determinism
        stack.make_contiguous().sort_unstable();

        while let Some(node_id) = stack.pop_back() {
            // If our attempts failed and we let a duplicate sneak onto the stack,
            // skip if if we've already evaluated it
            if visited.contains(&node_id) || !graph.contains_node(node_id) {
                continue;
            }

            // Add all the current node's inputs to the stack
            buffer.extend(graph.inputs(node_id).filter_map(|(_, input, _, _)| {
                let input_id = input.node_id();
                (!visited.contains(&input_id) && !stack.contains(&input_id)).then(|| input_id)
            }));

            // If there's inputs that are yet to be processed, push this node to the very
            // bottom of the stack and add all of its dependencies to the top of the stack
            if !buffer.is_empty()
                || graph
                    .inputs(node_id)
                    .any(|(_, input, ..)| !visited.contains(&input.node_id()))
            {
                // Sort for determinism
                buffer.sort_unstable();

                stack.reserve(buffer.len() + 1);
                stack.extend(buffer.drain(..));
                stack.push_front(node_id);

                continue;
            }

            // Visit the node
            self.visit(graph, node_id);

            // Add the node to the list of nodes we've already visited
            let existed = visited.insert(node_id);
            debug_assert!(existed);

            // If the node was deleted within the call to `.visit()` then don't add its dependents
            if graph.contains_node(node_id) {
                // Add all the current node's outputs to the stack
                buffer.extend(
                    graph
                        .outputs(node_id)
                        .filter_map(|(_, data)| data)
                        .filter_map(|(output, _, _)| {
                            let output_id = output.node_id();

                            // At first glance it may seem like `visited.contains(&output_id)` should always return
                            // true, it won't in the case of two identical edges to the same node
                            (!visited.contains(&output_id) && !stack.contains(&output_id))
                                .then(|| output_id)
                        }),
                );

                // Sort for determinism
                buffer.sort_unstable();
                stack.extend(buffer.drain(..));
            }
        }

        self.post_visit_graph(graph, &visited);
        self.did_change()
    }

    fn post_visit_graph(&mut self, _graph: &mut Rvsdg, _visited: &HashSet<NodeId>) {}

    fn visit(&mut self, graph: &mut Rvsdg, node_id: NodeId) {
        if let Some(node) = graph.try_node(node_id).cloned() {
            match node {
                Node::Int(int, value) => self.visit_int(graph, int, value),
                Node::Bool(bool, value) => self.visit_bool(graph, bool, value),
                Node::Add(add) => self.visit_add(graph, add),
                Node::Load(load) => self.visit_load(graph, load),
                Node::Store(store) => self.visit_store(graph, store),
                Node::Start(start) => self.visit_start(graph, start),
                Node::End(end) => self.visit_end(graph, end),
                Node::Input(input) => self.visit_input(graph, input),
                Node::Output(output) => self.visit_output(graph, output),
                Node::Theta(theta) => self.visit_theta(graph, theta),
                Node::Eq(eq) => self.visit_eq(graph, eq),
                Node::Not(not) => self.visit_not(graph, not),
                Node::Phi(phi) => self.visit_phi(graph, phi),
                Node::InputPort(input_param) => self.visit_input_param(graph, input_param),
                Node::OutputPort(output_param) => self.visit_output_param(graph, output_param),
            }
        } else {
            tracing::error!("visited node that doesn't exist: {:?}", node_id);
        }
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, _int: Int, _value: i32) {}
    fn visit_bool(&mut self, _graph: &mut Rvsdg, _bool: Bool, _value: bool) {}
    fn visit_add(&mut self, _graph: &mut Rvsdg, _add: Add) {}
    fn visit_load(&mut self, _graph: &mut Rvsdg, _load: Load) {}
    fn visit_store(&mut self, _graph: &mut Rvsdg, _store: Store) {}
    fn visit_start(&mut self, _graph: &mut Rvsdg, _start: Start) {}
    fn visit_end(&mut self, _graph: &mut Rvsdg, _end: End) {}
    fn visit_input(&mut self, _graph: &mut Rvsdg, _input: Input) {}
    fn visit_output(&mut self, _graph: &mut Rvsdg, _output: Output) {}
    fn visit_theta(&mut self, _graph: &mut Rvsdg, _theta: Theta) {}
    fn visit_eq(&mut self, _graph: &mut Rvsdg, _eq: Eq) {}
    fn visit_not(&mut self, _graph: &mut Rvsdg, _not: Not) {}
    fn visit_phi(&mut self, _graph: &mut Rvsdg, _phi: Phi) {}
    fn visit_input_param(&mut self, _graph: &mut Rvsdg, _input_param: InputParam) {}
    fn visit_output_param(&mut self, _graph: &mut Rvsdg, _output_param: OutputParam) {}
}
