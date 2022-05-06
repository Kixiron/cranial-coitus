use crate::{
    graph::{
        Add, Bool, Byte, EdgeKind, Eq, Gamma, InputParam, InputPort, Int, Mul, Neg, Neq, Node,
        NodeExt, NodeId, Not, OutputParam, OutputPort, Rvsdg, Sub, Theta,
    },
    ir::Const,
    passes::{
        utils::{BinaryOp, Changes, UnaryOp},
        Pass,
    },
    utils::{AssertNone, HashMap, HashSet},
    values::{Cell, Ptr},
};
use std::collections::{BTreeSet, VecDeque};

use super::utils::ChangeReport;

/// Loop invariant code motion
// TODO: Pull out expressions that only depend on invariant inputs
// TODO: Demote variant inputs to invariant ones where possible
pub struct Licm {
    changes: Changes<3>,
    within_theta: bool,
    invariant_exprs: HashSet<OutputPort>,
}

impl Licm {
    pub fn new() -> Self {
        Self {
            changes: Changes::new([
                "gamma-inputs",
                "hoisted-theta-exprs",
                "theta-invariant-variants",
            ]),
            within_theta: false,
            invariant_exprs: HashSet::with_hasher(Default::default()),
        }
    }

    fn within_theta(mut self, within_theta: bool) -> Self {
        self.within_theta = within_theta;
        self
    }

    fn input_invariant(&self, graph: &Rvsdg, input: InputPort) -> bool {
        self.invariant_exprs.contains(&graph.input_source(input))
    }

    #[allow(dead_code)]
    fn pull_out_constants(
        &mut self,
        graph: &mut Rvsdg,
        body: &mut Rvsdg,
        input_pairs: HashMap<OutputPort, OutputPort>,
        invariant_inputs: &[(OutputPort, OutputPort)],
        invariant_exprs: &HashSet<OutputPort>,
    ) -> Vec<(OutputPort, NodeId)> {
        // TODO: Buffers
        let mut invariant_exprs: VecDeque<_> = invariant_exprs.iter().copied().collect();
        invariant_exprs.make_contiguous().sort_unstable();

        let mut puller =
            ExpressionPuller::new(graph, body, invariant_inputs, invariant_exprs, input_pairs);

        while let Some(port) = puller.pop_invariant_expr() {
            let node_id = puller.body.port_parent(port);

            match *puller.body.get_node(node_id) {
                Node::Eq(eq) => puller.binary_op(port, eq),
                Node::Neq(neq) => puller.binary_op(port, neq),
                Node::Add(add) => puller.binary_op(port, add),
                Node::Sub(sub) => puller.binary_op(port, sub),
                Node::Mul(mul) => puller.binary_op(port, mul),

                Node::Not(not) => puller.unary_op(port, not),
                Node::Neg(neg) => puller.unary_op(port, neg),

                Node::Int(int, value) => puller.constant(int.value(), value),
                Node::Byte(byte, value) => puller.constant(byte.value(), value),
                Node::Bool(bool, value) => puller.constant(bool.value(), value),

                Node::InputParam(input) => puller.input_param(input),

                ref node => {
                    tracing::warn!("unhandled invariant node kind: {:?}", node);
                }
            }
        }

        puller.finish()
    }
}

// TODO: Invariant branches and loops
impl Pass for Licm {
    fn pass_name(&self) -> &str {
        "loop-invariant-code-motion"
    }

    fn did_change(&self) -> bool {
        self.changes.did_change()
    }

    fn report(&self) -> ChangeReport {
        self.changes.as_report()
    }

    fn reset(&mut self) {
        self.changes.reset();
        self.within_theta = false;
        self.invariant_exprs.clear();
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut visitor = Self::new().within_theta(true);
        visitor
            .invariant_exprs
            .extend(theta.invariant_input_params().map(|input| input.output()));

        let mut changed = visitor.visit_graph(theta.body_mut());
        self.changes.combine(&visitor.changes);

        // Pull out any constants that are within the loop body
        let invariant_inputs: Vec<_> = theta
            .invariant_input_pairs()
            .map(|(input, param)| (param.output(), graph.input_source(input)))
            .collect();

        let input_pairs = theta
            .input_pairs()
            .map(|(input, param)| (param.output(), graph.input_source(input)))
            .collect();

        let pulled_constants = self.pull_out_constants(
            graph,
            theta.body_mut(),
            input_pairs,
            &invariant_inputs,
            &visitor.invariant_exprs,
        );

        for (constant, param) in pulled_constants {
            let port = graph.create_value_input(theta.node());
            graph.add_value_edge(constant, port);
            theta.add_invariant_input_raw(port, param);

            self.changes.inc::<"hoisted-theta-exprs">();
            changed = true;
        }

        // If a variant input is a constant, make it invariant
        let outputs: Vec<_> = theta.output_pairs().collect();
        for (output, output_param) in outputs {
            if let Some(&input_param) = theta
                .body()
                .input_source_node(output_param.input())
                .as_input_param()
            {
                let input = theta
                    .variant_input_pair_ids()
                    .find_map(|(input, param)| (param == input_param.node()).then(|| input));

                if let Some(input) = input {
                    tracing::trace!(
                        "demoting variant input {:?} (feedback from {:?}) to an invariant input",
                        input,
                        output,
                    );

                    let source = graph.input_source(input);
                    graph.rewire_dependents(output, source);
                    theta.remove_variant_input(input);
                    theta.add_invariant_input_raw(input, input_param.node());

                    self.changes.inc::<"theta-invariant-variants">();
                    changed = true;
                }
            }
        }

        if changed {
            graph.replace_node(theta.node(), theta);
        }
    }

    // TODO: Pull branch invariant code from branches?
    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut visitor = Self::new();

        let mut changed = visitor.visit_graph(gamma.true_mut());
        self.changes.combine(&visitor.changes);
        visitor.reset();

        changed |= visitor.visit_graph(gamma.false_mut());
        self.changes.combine(&visitor.changes);

        let outputs: Vec<_> = gamma
            .outputs()
            .iter()
            .copied()
            .zip(gamma.output_params().iter().copied())
            .collect();

        for (output, [true_out, false_out]) in outputs {
            let (true_out, false_out) = (
                *gamma.true_branch().to_node::<OutputParam>(true_out),
                *gamma.false_branch().to_node::<OutputParam>(false_out),
            );

            let (true_in, false_in) = (
                gamma
                    .true_branch()
                    .input_source_node(true_out.input())
                    .as_input_param(),
                gamma
                    .false_branch()
                    .input_source_node(false_out.input())
                    .as_input_param(),
            );

            if let Some((true_in, false_in)) = true_in.zip(false_in) {
                let true_input = *gamma
                    .inputs()
                    .iter()
                    .zip(gamma.input_params())
                    .find(|(_, &[param, _])| param == true_in.node())
                    .unwrap()
                    .0;
                let false_input = *gamma
                    .inputs()
                    .iter()
                    .zip(gamma.input_params())
                    .find(|(_, &[_, param])| param == false_in.node())
                    .unwrap()
                    .0;

                if true_input == false_input {
                    let input = true_input;

                    tracing::trace!(
                        "hoisting gamma output {:?} (input from {:?}) to a direct dependency",
                        output,
                        input,
                    );

                    // Rewire dependents to consume directly from the source
                    let input_source = graph.input_source(input);
                    graph.rewire_dependents(output, input_source);

                    // Remove the output entries on the gamma node
                    let output_index = gamma
                        .outputs()
                        .iter()
                        .position(|&out| out == output)
                        .unwrap();
                    gamma.outputs_mut().remove(output_index);
                    gamma.output_params_mut().remove(output_index);

                    // Remove the output nodes from the gamma body
                    gamma.true_mut().remove_node(true_out.node());
                    gamma.false_mut().remove_node(false_out.node());

                    self.changes.inc::<"gamma-inputs">();
                }
            }
        }

        if changed {
            graph.replace_node(gamma.node(), gamma);
        }
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, _: Ptr) {
        if self.within_theta {
            self.invariant_exprs.insert(int.value());
        }
    }

    fn visit_byte(&mut self, _graph: &mut Rvsdg, byte: Byte, _: Cell) {
        if self.within_theta {
            self.invariant_exprs.insert(byte.value());
        }
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, _: bool) {
        if self.within_theta {
            self.invariant_exprs.insert(bool.value());
        }
    }

    fn visit_add(&mut self, graph: &mut Rvsdg, add: Add) {
        if self.within_theta
            && self.input_invariant(graph, add.rhs())
            && self.input_invariant(graph, add.lhs())
        {
            self.invariant_exprs.insert(add.value());
        }
    }

    fn visit_sub(&mut self, graph: &mut Rvsdg, sub: Sub) {
        if self.within_theta
            && self.input_invariant(graph, sub.rhs())
            && self.input_invariant(graph, sub.lhs())
        {
            self.invariant_exprs.insert(sub.value());
        }
    }

    fn visit_mul(&mut self, graph: &mut Rvsdg, mul: Mul) {
        if self.within_theta
            && self.input_invariant(graph, mul.rhs())
            && self.input_invariant(graph, mul.lhs())
        {
            self.invariant_exprs.insert(mul.value());
        }
    }

    fn visit_eq(&mut self, graph: &mut Rvsdg, eq: Eq) {
        if self.within_theta
            && self.input_invariant(graph, eq.rhs())
            && self.input_invariant(graph, eq.lhs())
        {
            self.invariant_exprs.insert(eq.value());
        }
    }

    fn visit_neq(&mut self, graph: &mut Rvsdg, neq: Neq) {
        if self.within_theta
            && self.input_invariant(graph, neq.rhs())
            && self.input_invariant(graph, neq.lhs())
        {
            self.invariant_exprs.insert(neq.value());
        }
    }

    fn visit_not(&mut self, graph: &mut Rvsdg, not: Not) {
        if self.within_theta && self.input_invariant(graph, not.input()) {
            self.invariant_exprs.insert(not.value());
        }
    }

    fn visit_neg(&mut self, graph: &mut Rvsdg, neg: Neg) {
        if self.within_theta && self.input_invariant(graph, neg.input()) {
            self.invariant_exprs.insert(neg.value());
        }
    }
}

impl Default for Licm {
    fn default() -> Self {
        Self::new()
    }
}

struct ExpressionPuller<'a> {
    graph: &'a mut Rvsdg,
    body: &'a mut Rvsdg,
    removals: BTreeSet<NodeId>,
    params: Vec<(OutputPort, NodeId)>,
    constants: HashMap<OutputPort, Const>,
    invariant_exprs: VecDeque<OutputPort>,
    input_pairs: HashMap<OutputPort, OutputPort>,
    param_to_new: HashMap<OutputPort, OutputPort>,
}

impl<'a> ExpressionPuller<'a> {
    fn new(
        graph: &'a mut Rvsdg,
        body: &'a mut Rvsdg,
        invariant_inputs: &[(OutputPort, OutputPort)],
        invariant_exprs: VecDeque<OutputPort>,
        input_pairs: HashMap<OutputPort, OutputPort>,
    ) -> Self {
        // TODO: Buffers
        let mut param_to_new = HashMap::with_capacity_and_hasher(
            invariant_inputs.len() + invariant_exprs.len(),
            Default::default(),
        );
        param_to_new.extend(invariant_inputs.iter().copied());

        Self {
            graph,
            body,
            removals: BTreeSet::new(),
            params: Vec::new(),
            constants: HashMap::default(),
            invariant_exprs,
            input_pairs,
            param_to_new,
        }
    }

    fn pop_invariant_expr(&mut self) -> Option<OutputPort> {
        self.invariant_exprs.pop_back()
    }

    fn finish(self) -> Vec<(OutputPort, NodeId)> {
        self.body.bulk_remove_nodes(&self.removals);
        self.params
    }

    fn constant<T>(&mut self, output: OutputPort, value: T)
    where
        T: Into<Const>,
    {
        self.constants.insert(output, value.into());
    }

    fn binary_op<T>(&mut self, port: OutputPort, node: T)
    where
        T: BinaryOp,
    {
        let lhs_source = self.body.input_source(node.lhs());
        let lhs = self.param_to_new.get(&lhs_source).copied();
        let lhs_const = self.constants.get(&lhs_source).copied();

        let rhs_source = self.body.input_source(node.rhs());
        let rhs = self.param_to_new.get(&rhs_source).copied();
        let rhs_const = self.constants.get(&rhs_source).copied();

        let (lhs, rhs) = match (lhs, lhs_const, rhs, rhs_const) {
            (Some(lhs), _, Some(rhs), _) => (lhs, rhs),

            (_, Some(lhs), Some(rhs), _) => {
                let lhs = self.graph.constant(lhs).value();
                self.param_to_new.insert(lhs_source, lhs);

                (lhs, rhs)
            }

            (Some(lhs), _, _, Some(rhs)) => {
                let rhs = self.graph.constant(rhs).value();
                self.param_to_new.insert(rhs_source, rhs);

                (lhs, rhs)
            }

            (_, Some(lhs), _, Some(rhs)) => {
                let lhs = self.graph.constant(lhs).value();
                self.param_to_new.insert(lhs_source, lhs);

                let rhs = self.graph.constant(rhs).value();
                self.param_to_new.insert(rhs_source, rhs);

                (lhs, rhs)
            }

            _ => {
                self.invariant_exprs.push_front(port);
                return;
            }
        };

        let input = self.body.input_param(EdgeKind::Value);
        self.body.rewire_dependents(node.value(), input.output());
        self.removals.insert(node.node());

        let new_node = T::make_in_graph(self.graph, lhs, rhs);
        self.params.push((new_node.value(), input.node()));
        self.param_to_new
            .insert(input.output(), new_node.value())
            .debug_unwrap_none();
    }

    fn unary_op<T>(&mut self, port: OutputPort, node: T)
    where
        T: UnaryOp,
    {
        let input_source = self.body.input_source(node.input());
        let input = if let Some(input) = self.param_to_new.get(&input_source).copied() {
            input
        } else if let Some(input) = self.constants.get(&input_source).copied() {
            let input = self.graph.constant(input).value();
            self.param_to_new.insert(input_source, input);
            input
        } else {
            self.invariant_exprs.push_front(port);
            return;
        };

        let param = self.body.input_param(EdgeKind::Value);
        self.body.rewire_dependents(node.value(), param.output());
        self.removals.insert(node.node());

        let new_node = T::make_in_graph(self.graph, input);
        self.params.push((new_node.value(), param.node()));
        self.param_to_new
            .insert(param.output(), new_node.value())
            .debug_unwrap_none();
    }

    fn input_param(&mut self, input: InputParam) {
        self.param_to_new
            .insert(input.output(), self.input_pairs[&input.output()])
            .debug_unwrap_none();
    }
}

test_opts! {
    move_constants_from_theta,
    passes = |_| bvec![Licm::new()],
    output = [],
    |graph, effect, tape_len| {
        graph.theta([], [], effect, |graph, effect, _invariant, _variant| {
            graph.int(Ptr::new(20, tape_len));
            let bool = graph.bool(false);

            ThetaData::new([], bool.value(), effect)
        })
        .output_effect()
        .unwrap()
    },
}
