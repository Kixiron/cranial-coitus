mod add_sub_loop;
mod associative_add;
mod const_folding;
mod dce;
mod eliminate_const_gamma;
mod expr_dedup;
mod licm;
mod mem2reg;
mod unobserved_store;
mod zero_loop;

pub use add_sub_loop::AddSubLoop;
pub use associative_add::AssociativeAdd;
pub use const_folding::ConstFolding;
pub use dce::Dce;
pub use eliminate_const_gamma::ElimConstGamma;
pub use expr_dedup::ExprDedup;
pub use licm::Licm;
pub use mem2reg::Mem2Reg;
pub use unobserved_store::UnobservedStore;
pub use zero_loop::ZeroLoop;

use crate::{
    graph::{
        Add, Bool, End, Eq, Gamma, Input, InputParam, Int, Load, Neg, Node, NodeId, Not, Output,
        OutputParam, Rvsdg, Start, Store, Theta,
    },
    utils::HashSet,
};
use std::collections::VecDeque;

// TODO:
// - Addition https://esolangs.org/wiki/Brainfuck_algorithms#x_.3D_x_.2B_y
// - Subtraction https://esolangs.org/wiki/Brainfuck_algorithms#x_.3D_x_-_y
// - Copy https://esolangs.org/wiki/Brainfuck_algorithms#x_.3D_y
// - Multiplication https://esolangs.org/wiki/Brainfuck_algorithms#x_.3D_x_.2A_y
// - Squared https://esolangs.org/wiki/Brainfuck_algorithms#x_.3D_x_.2A_x
// - Division https://esolangs.org/wiki/Brainfuck_algorithms#x_.3D_x_.2F_y
// - Xor https://esolangs.org/wiki/Brainfuck_algorithms#x_.3D_x_.5E_y
// - Swap https://esolangs.org/wiki/Brainfuck_algorithms#swap_x.2C_y
// - Negation https://esolangs.org/wiki/Brainfuck_algorithms#x_.3D_-x
// - Bitwise not https://esolangs.org/wiki/Brainfuck_algorithms#x_.3D_not_x_.28bitwise.29
// - Non-wrapping cell zeroing https://esolangs.org/wiki/Brainfuck_algorithms#Non-wrapping
// - The rest of the motifs here https://esolangs.org/wiki/Brainfuck_algorithms
// - Data flow propagation: Unconditional cell zeroing on zero branch
// - Technically speaking, if the program contains no `input()` or `output()` invocations
//   and does not infinitely loop it has no *observable* side effects and we can thusly
//   delete it all...
// - Redundant loads propagating through other effect kinds: This is probably best addressed by
//   separating the IO and memory effect kinds
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
// - Removing equivalent nodes (CSE)
// - Build dataflow lattices of variant theta values
// - Lower "find next cell with a value of `N`" to to a `memchr()` invocation
pub trait Pass {
    /// The name of the current pass
    fn pass_name(&self) -> &str;

    fn did_change(&self) -> bool;

    fn reset(&mut self);

    fn visit_graph(&mut self, graph: &mut Rvsdg) -> bool {
        let (mut stack, mut visited, mut buffer) = (
            VecDeque::with_capacity(graph.node_len() / 2),
            HashSet::with_capacity_and_hasher(graph.node_len(), Default::default()),
            Vec::new(),
        );

        self.visit_graph_inner(graph, &mut stack, &mut visited, &mut buffer)
    }

    fn visit_graph_inner(
        &mut self,
        graph: &mut Rvsdg,
        stack: &mut VecDeque<NodeId>,
        visited: &mut HashSet<NodeId>,
        buffer: &mut Vec<NodeId>,
    ) -> bool {
        visited.clear();
        buffer.clear();

        // Initialize the stack with all of the important nodes within the graph
        // The reason we do this weird pushing thing is to make sure that the start nodes
        // are to the end of the queue so that they'll be the first ones to be popped and
        // processed
        for node_id in graph.node_ids() {
            let node = graph.get_node(node_id);

            if node.is_start() || node.is_input_param() {
                stack.push_back(node_id);
            } else if node.is_end() || node.is_output_param() {
                stack.push_front(node_id);
            }
        }

        while let Some(node_id) = stack.pop_back() {
            // If our attempts failed and we let a duplicate sneak onto the stack,
            // skip if if we've already evaluated it
            if visited.contains(&node_id) || !graph.contains_node(node_id) {
                continue;
            }

            // Add all the current node's inputs to the stack
            let mut missing_inputs = false;
            buffer.extend(graph.try_inputs(node_id).filter_map(|(_, input)| {
                input.and_then(|(input, ..)| {
                    let input_id = input.node_id();

                    if !visited.contains(&input_id) {
                        missing_inputs = true;
                        Some(input_id)
                    } else {
                        None
                    }
                })
            }));

            // If there's inputs that are yet to be processed, push the current
            // node into the stack and then all of its dependencies, ensuring
            // that the dependencies are processed first
            if missing_inputs {
                stack.reserve(buffer.len() + 1);
                stack.push_back(node_id);
                stack.extend(buffer.drain(..));

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
                        .get_node(node_id)
                        .outputs()
                        .into_iter()
                        .flat_map(|output| graph.get_outputs(output))
                        .filter_map(|(output_node, ..)| {
                            let output_id = output_node.node_id();

                            // At first glance it may seem like `visited.contains(&output_id)` should always return
                            // true, it won't in the case of two identical edges to the same node
                            (!visited.contains(&output_id) && !stack.contains(&output_id))
                                .then(|| output_id)
                        }),
                );

                stack.extend(buffer.drain(..));
            }
        }

        self.post_visit_graph(graph, visited);
        stack.clear();
        buffer.clear();
        visited.clear();

        self.did_change()
    }

    fn post_visit_graph(&mut self, _graph: &mut Rvsdg, _visited: &HashSet<NodeId>) {}

    fn visit(&mut self, graph: &mut Rvsdg, node_id: NodeId) {
        // FIXME: These clones are really not good
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
                Node::Theta(theta) => self.visit_theta(graph, *theta),
                Node::Eq(eq) => self.visit_eq(graph, eq),
                Node::Not(not) => self.visit_not(graph, not),
                Node::Neg(neg) => self.visit_neg(graph, neg),
                Node::Gamma(gamma) => self.visit_gamma(graph, *gamma),
                Node::InputParam(input_param) => self.visit_input_param(graph, input_param),
                Node::OutputParam(output_param) => self.visit_output_param(graph, output_param),
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
    fn visit_neg(&mut self, _graph: &mut Rvsdg, _neg: Neg) {}
    fn visit_gamma(&mut self, _graph: &mut Rvsdg, _gamma: Gamma) {}
    fn visit_input_param(&mut self, _graph: &mut Rvsdg, _input_param: InputParam) {}
    fn visit_output_param(&mut self, _graph: &mut Rvsdg, _output_param: OutputParam) {}
}

// FIXME: Lower this to a `memchr()` invocation
test_opts! {
    // do {
    //     v1 := int 4
    //     v2 := add v211, v1
    //     v3 := load v2 // eff: e251, pred: e230
    //     v4 := int 0
    //     v5 := eq v3, v4
    //     v6 := not v5
    //     v7 := out v2
    // } while { v6 }
    memchr_loop,
    passes = [ZeroLoop::new()],
    output = [0],
    |graph, mut effect| {
        let mut ptr = graph.int(0).value();

        // Store a non-zero value to the current cell
        let not_zero = graph.int(255).value();
        let store = graph.store(ptr, not_zero, effect);
        effect = store.effect();

        // Create the theta node
        let theta = graph.theta([], [ptr], effect, |graph, mut effect, _invariant, variant| {
            let ptr = variant[0];

            let zero = graph.int(0).value();
            let four = graph.int(4).value();

            let add = graph.add(ptr, four);
            let load = graph.load(add.value(), effect);
            effect = load.effect();

            let not_eq_zero = graph.neq(load.value(), zero);

            ThetaData::new([ptr], not_eq_zero.value(), effect)
        });
        ptr = theta.outputs()[0];
        effect = theta.output_effect().unwrap();

        // Load the cell's value
        let load = graph.load(ptr, effect);
        effect = load.effect();

        // Output the value at the index (should be zero)
        let output = graph.output(load.value(), effect);
        output.effect()
    },
}
