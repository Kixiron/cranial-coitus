mod add_sub_loop;
mod associative_ops;
mod canonicalize;
mod const_folding;
mod copy_cell;
mod dataflow;
mod dce;
mod eliminate_const_gamma;
mod equality;
mod expr_dedup;
mod fold_arithmetic;
mod fuse_io;
mod licm;
mod mem2reg;
mod move_cell;
mod scan_loops;
mod shift;
mod square_cell;
mod symbolic_eval;
mod unobserved_store;
mod utils;
mod zero_loop;

pub use add_sub_loop::AddSubLoop;
pub use associative_ops::AssociativeOps;
pub use canonicalize::Canonicalize;
pub use const_folding::ConstFolding;
pub use copy_cell::CopyCell;
pub use dataflow::{Dataflow, DataflowSettings};
pub use dce::Dce;
pub use eliminate_const_gamma::ElimConstGamma;
pub use equality::Equality;
pub use expr_dedup::ExprDedup;
pub use fold_arithmetic::FoldArithmetic;
pub use fuse_io::FuseIO;
pub use licm::Licm;
pub use mem2reg::Mem2Reg;
pub use move_cell::MoveCell;
pub use scan_loops::ScanLoops;
pub use shift::ShiftCell;
pub use square_cell::SquareCell;
pub use symbolic_eval::SymbolicEval;
pub use unobserved_store::UnobservedStore;
pub use zero_loop::ZeroLoop;

use crate::{
    graph::{
        Add, Bool, Byte, End, Eq, Gamma, Input, InputParam, Int, Load, Mul, Neg, Neq, Node,
        NodeExt, NodeId, Not, Output, OutputParam, Rvsdg, Scan, Start, Store, Sub, Theta,
    },
    passes::utils::ChangeReport,
    utils::HashSet,
    values::{Cell, Ptr},
};
use std::{cell::RefCell, collections::VecDeque};

#[derive(Debug, Clone)]
pub struct PassConfig {
    tape_len: u16,
    tape_operations_wrap: bool,
    cell_operations_wrap: bool,
}

impl PassConfig {
    pub fn new(tape_len: u16, tape_operations_wrap: bool, cell_operations_wrap: bool) -> Self {
        Self {
            tape_len,
            tape_operations_wrap,
            cell_operations_wrap,
        }
    }
}

// TODO: Genetic algorithm for pass ordering
//       https://kunalspathak.github.io/2021-07-22-Genetic-Algorithms-In-LSRA/
pub fn default_passes(config: &PassConfig) -> Vec<Box<dyn Pass>> {
    let tape_len = config.tape_len;

    bvec![
        UnobservedStore::new(tape_len),
        ConstFolding::new(tape_len),
        FoldArithmetic::new(tape_len),
        AssociativeOps::new(tape_len),
        ZeroLoop::new(tape_len),
        Mem2Reg::new(tape_len),
        AddSubLoop::new(tape_len),
        ShiftCell::new(tape_len),
        FuseIO::new(),
        MoveCell::new(tape_len),
        ScanLoops::new(tape_len),
        Dce::new(),
        ElimConstGamma::new(),
        ConstFolding::new(tape_len),
        SquareCell::new(tape_len),
        SymbolicEval::new(tape_len),
        Licm::new(),
        CopyCell::new(tape_len),
        Equality::new(),
        ExprDedup::new(tape_len),
        // Dataflow::new(DataflowSettings::new(config)),
        Canonicalize::new(),
        Dce::new(),
    ]
}

thread_local! {
    /// A cache to hold buffers for visitors to reuse
    #[allow(clippy::type_complexity)]
    static VISIT_GRAPH_CACHE: RefCell<Vec<(VecDeque<NodeId>, HashSet<NodeId>, Vec<NodeId>)>> = RefCell::new(Vec::new());
}

// TODO:
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
pub trait Pass {
    /// The name of the current pass
    fn pass_name(&self) -> &str;

    fn did_change(&self) -> bool;

    fn reset(&mut self);

    fn report(&self) -> ChangeReport {
        ChangeReport::default()
    }

    fn visit_graph(&mut self, graph: &mut Rvsdg) -> bool {
        // Attempt to reuse any available buffers
        let (mut stack, mut visited, mut buffer) = VISIT_GRAPH_CACHE
            .with(|buffers| buffers.borrow_mut().pop())
            // If we couldn't reuse a buffer, make new ones
            .unwrap_or_else(|| {
                (
                    VecDeque::with_capacity(graph.node_len() / 2),
                    HashSet::with_capacity_and_hasher(graph.node_len(), Default::default()),
                    Vec::new(),
                )
            });

        let result = self.visit_graph_inner(graph, &mut stack, &mut visited, &mut buffer);

        // Clear the buffers before inserting them into the cache
        stack.clear();
        visited.clear();
        buffer.clear();
        VISIT_GRAPH_CACHE.with(|buffers| buffers.borrow_mut().push((stack, visited, buffer)));

        result
    }

    // TODO: We really want to be doing post-ordered construction of the work list,
    //       going from the end nodes backwards to the start nodes. However, it's
    //       difficult since graph manipulations would require us to constantly
    //       rebuild the traversal tree to include the new nodes
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
                    let input_id = input.node();

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
            self.after_visit(graph, node_id);

            // Add the node to the list of nodes we've already visited
            let didnt_exist = visited.insert(node_id);
            debug_assert!(didnt_exist);

            // If the node was deleted within the call to `.visit()` then don't add its dependents
            if graph.contains_node(node_id) {
                // Add all the current node's outputs to the stack
                buffer.extend(
                    graph
                        .get_node(node_id)
                        .all_output_ports()
                        .into_iter()
                        .flat_map(|output| graph.get_outputs(output))
                        .filter_map(|(output_node, ..)| {
                            let output_id = output_node.node();

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

    fn after_visit(&mut self, _graph: &mut Rvsdg, _node_id: NodeId) {}

    fn visit(&mut self, graph: &mut Rvsdg, node_id: NodeId) {
        // FIXME: These clones are really not good
        if let Some(node) = graph.try_node(node_id).cloned() {
            match node {
                Node::Int(int, value) => self.visit_int(graph, int, value),
                Node::Byte(byte, value) => self.visit_byte(graph, byte, value),
                Node::Bool(bool, value) => self.visit_bool(graph, bool, value),
                Node::Add(add) => self.visit_add(graph, add),
                Node::Sub(sub) => self.visit_sub(graph, sub),
                Node::Mul(mul) => self.visit_mul(graph, mul),
                Node::Load(load) => self.visit_load(graph, load),
                Node::Store(store) => self.visit_store(graph, store),
                Node::Scan(scan) => self.visit_scan(graph, scan),
                Node::Start(start) => self.visit_start(graph, start),
                Node::End(end) => self.visit_end(graph, end),
                Node::Input(input) => self.visit_input(graph, input),
                Node::Output(output) => self.visit_output(graph, output),
                Node::Theta(theta) => self.visit_theta(graph, *theta),
                Node::Eq(eq) => self.visit_eq(graph, eq),
                Node::Neq(neq) => self.visit_neq(graph, neq),
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

    fn visit_int(&mut self, _graph: &mut Rvsdg, _int: Int, _value: Ptr) {}
    fn visit_byte(&mut self, _graph: &mut Rvsdg, _byte: Byte, _value: Cell) {}
    fn visit_bool(&mut self, _graph: &mut Rvsdg, _bool: Bool, _value: bool) {}

    fn visit_add(&mut self, _graph: &mut Rvsdg, _add: Add) {}
    fn visit_sub(&mut self, _graph: &mut Rvsdg, _sub: Sub) {}
    fn visit_mul(&mut self, _graph: &mut Rvsdg, _mul: Mul) {}

    fn visit_not(&mut self, _graph: &mut Rvsdg, _not: Not) {}
    fn visit_neg(&mut self, _graph: &mut Rvsdg, _neg: Neg) {}

    fn visit_eq(&mut self, _graph: &mut Rvsdg, _eq: Eq) {}
    fn visit_neq(&mut self, _graph: &mut Rvsdg, _neq: Neq) {}

    fn visit_load(&mut self, _graph: &mut Rvsdg, _load: Load) {}
    fn visit_store(&mut self, _graph: &mut Rvsdg, _store: Store) {}

    fn visit_scan(&mut self, _graph: &mut Rvsdg, _scan: Scan) {}
    fn visit_input(&mut self, _graph: &mut Rvsdg, _input: Input) {}
    fn visit_output(&mut self, _graph: &mut Rvsdg, _output: Output) {}

    // TODO: To somewhat address the cloning problem we could have two methods,
    //       one that takes an immutable reference to both the graph and the node
    //       and one that gives a mutable reference to the graph and the node's id
    // fn visit_theta(&mut self, _graph: &Rvsdg, _theta: &Theta) -> Option<bool> {
    //     Some(true)
    // }
    // fn mutate_theta(&mut self, _graph: &mut Rvsdg, _theta: NodeId) {}
    fn visit_theta(&mut self, _graph: &mut Rvsdg, _theta: Theta) {}
    fn visit_gamma(&mut self, _graph: &mut Rvsdg, _gamma: Gamma) {}

    fn visit_input_param(&mut self, _graph: &mut Rvsdg, _input: InputParam) {}
    fn visit_output_param(&mut self, _graph: &mut Rvsdg, _output: OutputParam) {}

    fn visit_start(&mut self, _graph: &mut Rvsdg, _start: Start) {}
    fn visit_end(&mut self, _graph: &mut Rvsdg, _end: End) {}
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
    passes = |tape_len| bvec![ZeroLoop::new(tape_len)],
    output = [0],
    |graph, mut effect, tape_len| {
        let mut ptr = graph.int(Ptr::zero(tape_len)).value();

        // Store a non-zero value to the current cell
        let not_zero = graph.int(Ptr::new(255, tape_len)).value();
        let store = graph.store(ptr, not_zero, effect);
        effect = store.output_effect();

        // Create the theta node
        let theta = graph.theta([], [ptr], effect, |graph, mut effect, _invariant, variant| {
            let ptr = variant[0];

            let zero = graph.int(Ptr::zero(tape_len)).value();
            let four = graph.int(Ptr::new(4, tape_len)).value();

            let add = graph.add(ptr, four);
            let load = graph.load(add.value(), effect);
            effect = load.output_effect();

            let not_eq_zero = graph.neq(load.output_value(), zero);

            ThetaData::new([ptr], not_eq_zero.value(), effect)
        });
        ptr = theta.output_ports().next().unwrap();
        effect = theta.output_effect().unwrap();

        // Load the cell's value
        let load = graph.load(ptr, effect);
        effect = load.output_effect();

        // Output the value at the index (should be zero)
        let output = graph.output(load.output_value(), effect);
        output.output_effect()
    },
}
