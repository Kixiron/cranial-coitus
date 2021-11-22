use crate::{
    graph::{
        Add, Bool, EdgeKind, Eq, Gamma, InputParam, InputPort, Int, Mul, Neg, NodeExt, Not,
        OutputPort, Rvsdg, Theta,
    },
    ir::Const,
    passes::Pass,
    utils::AssertNone,
};
use std::collections::BTreeMap;

/// Evaluates constant operations within the program
pub struct ConstFolding {
    values: BTreeMap<OutputPort, Const>,
    changed: bool,
}

impl ConstFolding {
    pub fn new() -> Self {
        Self {
            values: BTreeMap::new(),
            changed: false,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    fn operand(&self, graph: &Rvsdg, input: InputPort) -> (OutputPort, Option<i32>) {
        let source = graph.input_source(input);
        let value = self.values.get(&source).and_then(Const::convert_to_i32);

        (source, value)
    }
}

// TODO: Double bitwise and logical negation
impl Pass for ConstFolding {
    fn pass_name(&self) -> &str {
        "constant-folding"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.values.clear();
        self.changed = false;
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: i32) {
        let replaced = self.values.insert(int.value(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Int(value)));
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        let replaced = self.values.insert(bool.value(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Bool(value)));
    }

    fn visit_add(&mut self, graph: &mut Rvsdg, add: Add) {
        let inputs = [
            self.operand(graph, add.lhs()),
            self.operand(graph, add.rhs()),
        ];

        match inputs {
            // If both sides of the add are known, we can evaluate it directly
            // `10 + 10 => 20`
            [(_, Some(lhs)), (_, Some(rhs))] => {
                let sum = lhs + rhs;
                tracing::debug!(lhs, rhs, "evaluated add {:?} to {}", add, sum);

                let int = graph.int(sum);
                graph.rewire_dependents(add.value(), int.value());
                self.changed();

                // Add the derived values to the known constants
                self.values.insert(int.value(), sum.into());
                self.values.insert(add.value(), sum.into());
            }

            // If either side of the add is zero, we can simplify it to the non-zero value
            // `x + 0 => x`, `0 + x => x`
            [(_, Some(0)), (value, None)] | [(value, None), (_, Some(0))] => {
                tracing::debug!(
                    "removing an addition by zero {:?} into a direct value of {:?}",
                    add,
                    value,
                );

                graph.rewire_dependents(add.value(), value);
                self.changed();

                // Add the derived value for the add node to the known constants
                if let Some(value) = self.values.get(&value).cloned() {
                    self.values.insert(add.value(), value);
                }
            }

            // If one side of the add is the negative of the other, the add simplifies into a zero
            // `x + -x => 0`, `-x + x => 0`
            [(lhs, _), (rhs, _)] | [(rhs, _), (lhs, _)]
                if graph
                    .cast_output_dest::<Neg>(rhs)
                    .map_or(false, |neg| graph.input_source(neg.input()) == lhs) =>
            {
                tracing::debug!(
                    ?add,
                    "removing an addition by negated value into a direct value of 0",
                );

                let zero = graph.int(0);
                graph.rewire_dependents(add.value(), zero.value());
                self.changed();

                // Add the derived values to the known constants
                self.values.insert(zero.value(), 0.into());
                self.values.insert(add.value(), 0.into());
            }

            _ => {}
        }
    }

    fn visit_mul(&mut self, graph: &mut Rvsdg, mul: Mul) {
        let inputs = [
            self.operand(graph, mul.lhs()),
            self.operand(graph, mul.rhs()),
        ];

        match inputs {
            // If both sides of the multiply are known, we can evaluate it directly
            // `10 * 10 => 100`
            [(_, Some(lhs)), (_, Some(rhs))] => {
                let product = lhs * rhs;
                tracing::debug!(lhs, rhs, "evaluated multiply {:?} to {}", mul, product);

                let int = graph.int(product);
                graph.rewire_dependents(mul.value(), int.value());
                self.changed();

                // Add the derived values to the known constants
                self.values.insert(int.value(), product.into());
                self.values.insert(mul.value(), product.into());
            }

            // If either side of the multiply is zero, we can remove the multiply entirely for zero
            // `x * 0 => 0`, `0 * x => 0`
            [(zero, Some(0)), _] | [_, (zero, Some(0))] => {
                tracing::debug!(
                    zero_port = ?zero,
                    "removing an multiply by zero {:?} into a direct value of 0",
                    mul,
                );

                graph.rewire_dependents(mul.value(), zero);
                self.changed();

                // Add the derived value to the known constants
                self.values.insert(mul.value(), 0.into());
            }

            // If either side of the multiply is one, we can remove the multiply entirely for the non-one value
            // `x * 1 => x`, `1 * x => x`
            [(_, Some(1)), (value, None)] | [(value, None), (_, Some(1))] => {
                tracing::debug!(
                    "removing an multiply by one {:?} into a direct value of {:?}",
                    mul,
                    value,
                );

                graph.rewire_dependents(mul.value(), value);
                self.changed();

                // Add the derived value for the mul node to the known constants
                if let Some(value) = self.values.get(&value).cloned() {
                    self.values.insert(mul.value(), value);
                }
            }

            _ => {}
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut changed = false;
        let mut visitor = Self::new();

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        // Note: We only propagate **invariant** inputs into the loop, propagating
        //       variant inputs requires dataflow information
        for (input, param) in theta.invariant_input_pairs() {
            if let Some(constant) = self.values.get(&graph.input_source(input)).cloned() {
                let replaced = visitor.values.insert(param.output(), constant);
                debug_assert!(replaced.is_none());
            }
        }

        changed |= visitor.visit_graph(theta.body_mut());

        if changed {
            graph.replace_node(theta.node(), theta);
            self.changed();
        }
    }

    fn visit_eq(&mut self, graph: &mut Rvsdg, eq: Eq) {
        let [(lhs_source, lhs), (rhs_source, rhs)] =
            [self.operand(graph, eq.lhs()), self.operand(graph, eq.rhs())];

        // If both values are known we can statically evaluate the comparison
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            tracing::debug!(
                "replaced const eq with {} ({:?} == {:?}) {:?} ({:?} == {:?})",
                lhs == rhs,
                lhs,
                rhs,
                eq,
                graph.get_output(lhs_source),
                graph.get_output(rhs_source),
            );

            let are_equal = graph.bool(lhs == rhs);
            self.values
                .insert(are_equal.value(), Const::Bool(lhs == rhs));
            self.values.remove(&eq.value());

            graph.rewire_dependents(eq.value(), are_equal.value());

            self.changed();

        // If the operands are equal this comparison will always be true
        } else if lhs_source == rhs_source {
            tracing::debug!(
                "replaced self-equality with true ({:?} == {:?}) {:?}",
                lhs_source,
                rhs_source,
                eq,
            );

            let true_val = graph.bool(true);
            self.values.insert(true_val.value(), Const::Bool(true));
            self.values.remove(&eq.value());

            graph.rewire_dependents(eq.value(), true_val.value());

            self.changed();
        }
    }

    fn visit_not(&mut self, graph: &mut Rvsdg, not: Not) {
        let (_, output, edge) = graph.get_input(not.input());
        debug_assert_eq!(edge, EdgeKind::Value);

        if let Some(value) = self.values.get(&output).and_then(Const::as_bool) {
            tracing::debug!("constant folding 'not {}' to '{}'", value, !value);

            let inverted = graph.bool(!value);
            self.values.insert(inverted.value(), Const::Bool(!value));
            self.values.remove(&not.value());

            graph.rewire_dependents(not.value(), inverted.value());

            self.changed();
        }
    }

    fn visit_neg(&mut self, graph: &mut Rvsdg, neg: Neg) {
        let (_, output, edge) = graph.get_input(neg.input());
        debug_assert_eq!(edge, EdgeKind::Value);

        if let Some(value) = self.values.get(&output) {
            tracing::debug!("constant folding 'neg {}' to '{}'", value, !value);

            let inverted = match -value {
                Const::Int(int) => graph.int(int).value(),
                // FIXME: Do we need a byte node?
                Const::U8(byte) => graph.int(byte as i32).value(),
                Const::Bool(bool) => graph.bool(bool).value(),
            };
            self.values.remove(&neg.value());

            graph.rewire_dependents(neg.value(), inverted);

            self.changed();
        }
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;
        let (mut truthy_visitor, mut falsy_visitor) = (Self::new(), Self::new());

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[truthy_param, falsy_param]) in
            gamma.inputs().iter().zip(gamma.input_params())
        {
            let (_, output, _) = graph.get_input(input);

            if let Some(constant) = self.values.get(&output).copied() {
                let true_param = gamma.true_branch().to_node::<InputParam>(truthy_param);
                truthy_visitor
                    .values
                    .insert(true_param.output(), constant)
                    .debug_unwrap_none();

                let false_param = gamma.false_branch().to_node::<InputParam>(falsy_param);
                falsy_visitor
                    .values
                    .insert(false_param.output(), constant)
                    .debug_unwrap_none();
            }
        }

        changed |= truthy_visitor.visit_graph(gamma.true_mut());
        changed |= falsy_visitor.visit_graph(gamma.false_mut());

        for (&port, &param) in gamma.outputs().iter().zip(gamma.output_params()) {
            let true_output = gamma
                .true_branch()
                .get_input(
                    gamma
                        .true_branch()
                        .get_node(param[0])
                        .to_output_param()
                        .input(),
                )
                .1;

            let false_output = gamma
                .false_branch()
                .get_input(
                    gamma
                        .false_branch()
                        .get_node(param[1])
                        .to_output_param()
                        .input(),
                )
                .1;

            if let (Some(truthy), Some(falsy)) = (
                truthy_visitor.values.get(&true_output).cloned(),
                falsy_visitor.values.get(&false_output).cloned(),
            ) {
                if truthy == falsy {
                    tracing::trace!("propagating {:?} out of gamma node", truthy);
                    self.values.insert(port, truthy);
                } else {
                    tracing::debug!(
                        "failed to propagate value out of gamma node, branches disagree ({:?} vs. {:?})",
                        truthy,
                        falsy,
                    );
                }
            }
        }

        if changed {
            graph.replace_node(gamma.node(), gamma);
            self.changed();
        }
    }
}

impl Default for ConstFolding {
    fn default() -> Self {
        Self::new()
    }
}

// TODO: Make sure that constants are propagated into gammas and thetas
//       as well as out of gammas
test_opts! {
    constant_add,
    passes = [ConstFolding::new(), Dce::new()],
    output = [30],
    |graph, effect| {
        let lhs = graph.int(10);
        let rhs = graph.int(20);
        let sum = graph.add(lhs.value(), rhs.value());

        graph.output(sum.value(), effect).output_effect()
    },
}

test_opts! {
    constant_mul,
    passes = [ConstFolding::new(), Dce::new()],
    output = [100],
    |graph, effect| {
        let lhs = graph.int(10);
        let rhs = graph.int(10);
        let sum = graph.mul(lhs.value(), rhs.value());

        graph.output(sum.value(), effect).output_effect()
    },
}

test_opts! {
    constant_sub,
    passes = [ConstFolding::new(), Dce::new()],
    output = [245],
    |graph, effect| {
        let lhs = graph.int(10);
        let rhs = graph.int(-20);
        let sum = graph.add(lhs.value(), rhs.value());

        graph.output(sum.value(), effect).output_effect()
    },
}

test_opts! {
    chained_booleans,
    passes = [ConstFolding::new(), Dce::new()],
    output = [1],
    |graph, effect| {
        let t = graph.bool(true);
        let f = graph.bool(false);

        let not1 = graph.not(t.value());
        let not2 = graph.not(not1.value());
        let eq1 = graph.eq(not2.value(), f.value());
        let eq2 = graph.eq(eq1.value(), not1.value());
        let eq3 = graph.eq(f.value(), eq2.value());
        let not3 = graph.not(eq3.value());

        graph.output(not3.value(), effect).output_effect()
    },
}
