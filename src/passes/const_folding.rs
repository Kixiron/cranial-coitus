use crate::{
    graph::{
        Add, Bool, Byte, EdgeKind, Eq, Gamma, InputParam, InputPort, Int, Mul, Neg, Neq, NodeExt,
        Not, OutputPort, Rvsdg, Sub, Theta,
    },
    ir::Const,
    passes::{
        utils::{ChangeReport, Changes, ConstantStore},
        Pass, PassConfig,
    },
    values::{Cell, Ptr},
};

/// Evaluates constant operations within the program
pub struct ConstFolding {
    values: ConstantStore,
    changes: Changes<5>,
    tape_len: u16,
}

impl ConstFolding {
    pub fn new(tape_len: u16) -> Self {
        Self {
            values: ConstantStore::new(tape_len),
            changes: Changes::new([
                "exprs-folded",
                "const-gamma-output",
                "propagated-inputs",
                "invariant-theta-output",
                "invariant-theta-feedback",
            ]),
            tape_len,
        }
    }

    fn operand(&self, graph: &Rvsdg, input: InputPort) -> (OutputPort, Option<Const>) {
        let source = graph.input_source(input);
        let value = self.values.get(source);

        (source, value)
    }
}

// TODO: Double bitwise and logical negation
impl Pass for ConstFolding {
    fn pass_name(&self) -> &'static str {
        "constant-folding"
    }

    fn did_change(&self) -> bool {
        self.changes.did_change()
    }

    fn reset(&mut self) {
        self.values.clear();
        self.changes.reset();
    }

    fn report(&self) -> ChangeReport {
        self.changes.as_report()
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: Ptr) {
        self.values.add(int.value(), value);
    }

    fn visit_byte(&mut self, _graph: &mut Rvsdg, byte: Byte, value: Cell) {
        self.values.add(byte.value(), value);
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        self.values.add(bool.value(), value);
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
                tracing::debug!(%lhs, %rhs, "evaluated add {:?} to {}", add, sum);

                let int = graph.constant(sum);
                graph.rewire_dependents(add.value(), int.value());

                // Add the derived values to the known constants
                self.values.add(int.value(), sum);
                self.values.add(add.value(), sum);

                self.changes.inc::<"exprs-folded">();
            }

            // If either side of the add is zero, we can simplify it to the non-zero value
            // `x + 0 => x`, `0 + x => x`
            [(_, Some(zero)), (value, None)] | [(value, None), (_, Some(zero))]
                if zero.is_zero() =>
            {
                tracing::debug!(
                    "removing an addition by zero {:?} into a direct value of {:?}",
                    add,
                    value,
                );

                graph.rewire_dependents(add.value(), value);

                // Add the derived value for the add node to the known constants
                if let Some(value) = self.values.get(value) {
                    self.values.add(add.value(), value);
                }

                self.changes.inc::<"exprs-folded">();
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

                let zero = graph.int(Ptr::zero(self.tape_len));
                graph.rewire_dependents(add.value(), zero.value());

                // Add the derived values to the known constants
                self.values.add(zero.value(), Ptr::zero(self.tape_len));
                self.values.add(add.value(), Ptr::zero(self.tape_len));

                self.changes.inc::<"exprs-folded">();
            }

            _ => {}
        }
    }

    fn visit_sub(&mut self, graph: &mut Rvsdg, sub: Sub) {
        let inputs = [
            self.operand(graph, sub.lhs()),
            self.operand(graph, sub.rhs()),
        ];

        match inputs {
            // If both sides of the sub are known, we can evaluate it directly
            // `10 - 10 => 20`
            [(_, Some(lhs)), (_, Some(rhs))] => {
                let difference = lhs - rhs;
                tracing::debug!(%lhs, %rhs, "evaluated sub {:?} to {}", sub, difference);

                let int = graph.constant(difference);
                graph.rewire_dependents(sub.value(), int.value());

                // Add the derived values to the known constants
                self.values.add(int.value(), difference);
                self.values.add(sub.value(), difference);

                self.changes.inc::<"exprs-folded">();
            }

            // If either side of the sub are zero, we can simplify it to the non-zero value
            // `x - 0 => x`
            [(value, None), (_, Some(zero))] if zero.is_zero() => {
                tracing::debug!(
                    "removing an subtraction by zero {:?} into a direct value of {:?}",
                    sub,
                    value,
                );

                graph.rewire_dependents(sub.value(), value);

                // Add the derived value for the sub node to the known constants
                if let Some(value) = self.values.get(value) {
                    self.values.add(sub.value(), value);
                }

                self.changes.inc::<"exprs-folded">();
            }

            // `0 - x => -x`
            [(_, Some(zero)), (value, None)] if zero.is_zero() => {
                tracing::debug!(
                    "removing an subtraction by zero {:?} (0 - x) into a -{:?}",
                    sub,
                    value,
                );

                let neg = graph.neg(value);
                graph.rewire_dependents(sub.value(), neg.value());

                // Add the derived value for the sub node to the known constants
                if let Some(value) = self.values.get(value) {
                    self.values.add(sub.value(), -value);
                    self.values.add(neg.value(), -value);
                }

                self.changes.inc::<"exprs-folded">();
            }

            // If one side of the sub is the negative of the other, the sub simplifies into a zero
            // `x - -x => 0`, `-x - x => 0`
            [(lhs, _), (rhs, _)] | [(rhs, _), (lhs, _)]
                if graph
                    .cast_output_dest::<Neg>(rhs)
                    .map_or(false, |neg| graph.input_source(neg.input()) == lhs) =>
            {
                tracing::debug!(
                    ?sub,
                    "removing an subtraction by negated value into a direct value of 0",
                );

                let zero = graph.int(Ptr::zero(self.tape_len));
                graph.rewire_dependents(sub.value(), zero.value());

                // Add the derived values to the known constants
                self.values.add(zero.value(), Ptr::zero(self.tape_len));
                self.values.add(sub.value(), Ptr::zero(self.tape_len));

                self.changes.inc::<"exprs-folded">();
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
                tracing::debug!(%lhs, %rhs, "evaluated multiply {:?} to {}", mul, product);

                let int = graph.constant(product);
                graph.rewire_dependents(mul.value(), int.value());

                // Add the derived values to the known constants
                self.values.add(int.value(), product);
                self.values.add(mul.value(), product);

                self.changes.inc::<"exprs-folded">();
            }

            // If either side of the multiply is zero, we can remove the multiply entirely for zero
            // `x * 0 => 0`, `0 * x => 0`
            [(zero, Some(zero_ptr)), _] | [_, (zero, Some(zero_ptr))] if zero_ptr.is_zero() => {
                tracing::debug!(
                    zero_port = ?zero,
                    "removing an multiply by zero {:?} into a direct value of 0",
                    mul,
                );

                graph.rewire_dependents(mul.value(), zero);

                // Add the derived value to the known constants
                self.values.add(mul.value(), Cell::zero());

                self.changes.inc::<"exprs-folded">();
            }

            // If either side of the multiply is one, we can remove the multiply entirely for the non-one value
            // `x * 1 => x`, `1 * x => x`
            [(_, Some(one)), (value, None)] | [(value, None), (_, Some(one))] if one.is_one() => {
                tracing::debug!(
                    "removing an multiply by one {:?} into a direct value of {:?}",
                    mul,
                    value,
                );

                graph.rewire_dependents(mul.value(), value);

                // Add the derived value for the mul node to the known constants
                if let Some(value) = self.values.get(value) {
                    self.values.add(mul.value(), value);
                }

                self.changes.inc::<"exprs-folded">();
            }

            _ => {}
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut changed = false;
        let mut visitor = Self::new(self.tape_len);

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        // Note: We only propagate **invariant** inputs into the loop, propagating
        //       variant inputs requires dataflow information
        let invariant_inputs: Vec<_> = theta.invariant_input_pairs().collect();
        for (input, param) in invariant_inputs {
            if let Some(constant) = self.values.get(graph.input_source(input))
                && !graph.input_source_node(input).is_constant()
            {
                let value = theta.body_mut().constant(constant).value();
                visitor.values.add(value, constant);
                theta.remove_invariant_input(input);
                theta.body_mut().rewire_dependents(param.output(), value);

                self.changes.inc::<"propagated-inputs">();
                changed = true;
            }
        }

        changed |= visitor.visit_graph(theta.body_mut());
        self.changes.combine(&visitor.changes);

        // Deduplicate variant inputs with identical values (currently only for constants)
        // and pull constant outputs out of the theta's body
        let variant_inputs: Vec<_> = theta
            .variant_input_pairs()
            .zip(theta.output_pairs())
            .collect();
        for ((input, input_param), (output, output_param)) in variant_inputs {
            let output_source = theta.body().input_source(output_param.input());
            if let Some(feedback_value) = visitor.values.get(output_source) {
                // If the input and feedback values are identical, deduplicate them
                if let Some(input_value) = self.values.get(graph.input_source(input))
                    && input_value == feedback_value
                {
                    // Rewire dependents on the parameters to the constant value and
                    // remove the parameters from the theta
                    theta.body_mut().rewire_dependents(input_param.output(), output_source);
                    theta.body_mut().remove_node(output_param.node());
                    theta.remove_variant_input(input);

                    self.changes.inc::<"invariant-theta-feedback">();
                }

                if graph.total_output_consumers(output) != 0 {
                    // Rewire any dependents of the output port to the constant value
                    let constant = graph.constant(feedback_value);
                    self.values.add(constant.value(), feedback_value);
                    graph.rewire_dependents(output, constant.value());

                    self.changes.inc::<"invariant-theta-output">();
                    changed = true;
                }
            }
        }

        if changed {
            graph.replace_node(theta.node(), theta);
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
            self.values.add(are_equal.value(), lhs == rhs);
            self.values.remove(eq.value());

            graph.rewire_dependents(eq.value(), are_equal.value());

            self.changes.inc::<"exprs-folded">();

        // If the operands are equal this comparison will always be true
        } else if lhs_source == rhs_source {
            tracing::debug!(
                "replaced self-equality with true ({:?} == {:?}) {:?}",
                lhs_source,
                rhs_source,
                eq,
            );

            let true_val = graph.bool(true);
            self.values.add(true_val.value(), true);
            self.values.remove(eq.value());

            graph.rewire_dependents(eq.value(), true_val.value());

            self.changes.inc::<"exprs-folded">();
        }
    }

    fn visit_neq(&mut self, graph: &mut Rvsdg, neq: Neq) {
        let [(lhs_source, lhs), (rhs_source, rhs)] = [
            self.operand(graph, neq.lhs()),
            self.operand(graph, neq.rhs()),
        ];

        // If both values are known we can statically evaluate the comparison
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let result = lhs != rhs;
            tracing::debug!(
                "replaced const neq with {} ({:?} != {:?}) {:?} ({:?} == {:?})",
                result,
                lhs,
                rhs,
                neq,
                graph.get_output(lhs_source),
                graph.get_output(rhs_source),
            );

            let are_inequal = graph.bool(result);
            self.values.add(are_inequal.value(), result);
            self.values.remove(neq.value());

            graph.rewire_dependents(neq.value(), are_inequal.value());

            self.changes.inc::<"exprs-folded">();

        // If the operands are equal this comparison will always be false
        } else if lhs_source == rhs_source {
            tracing::debug!(
                "replaced self-inequality with false ({:?} != {:?}) {:?}",
                lhs_source,
                rhs_source,
                neq,
            );

            let false_val = graph.bool(false);
            self.values.add(false_val.value(), false);
            self.values.remove(neq.value());

            graph.rewire_dependents(neq.value(), false_val.value());

            self.changes.inc::<"exprs-folded">();
        }
    }

    fn visit_not(&mut self, graph: &mut Rvsdg, not: Not) {
        let (_, output, edge) = graph.get_input(not.input());
        debug_assert_eq!(edge, EdgeKind::Value);

        if let Some(value) = self.values.bool(output) {
            tracing::debug!("constant folding 'not {}' to '{}'", value, !value);

            let inverted = graph.bool(!value);
            self.values.add(inverted.value(), !value);
            self.values.remove(not.value());

            graph.rewire_dependents(not.value(), inverted.value());

            self.changes.inc::<"exprs-folded">();
        }
    }

    fn visit_neg(&mut self, graph: &mut Rvsdg, neg: Neg) {
        let output = graph.input_source(neg.input());

        if let Some(value) = self.values.get(output) {
            tracing::debug!("constant folding 'neg {}' to '{}'", value, !value);

            let inverted = graph.constant(-value).value();
            self.values.remove(neg.value());
            self.values.add(inverted, -value);

            graph.rewire_dependents(neg.value(), inverted);

            self.changes.inc::<"exprs-folded">();
        }
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;
        let (mut true_visitor, mut false_visitor) =
            (Self::new(self.tape_len), Self::new(self.tape_len));

        let inputs: Vec<_> = gamma
            .inputs()
            .iter()
            .copied()
            .zip(gamma.input_params().iter().copied())
            .collect();

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (input, [truthy_param, falsy_param]) in inputs {
            let output = graph.input_source(input);

            // If the param has a constant value, pull the constant into both bodies
            // and allow dce to remove any redundant parameters or constants
            if let Some(constant) = self.values.get(output) {
                let (true_val, false_val) = (
                    gamma.true_mut().constant(constant).value(),
                    gamma.false_mut().constant(constant).value(),
                );

                true_visitor.values.add(true_val, constant);
                false_visitor.values.add(false_val, constant);

                let true_output = gamma
                    .true_branch()
                    .to_node::<InputParam>(truthy_param)
                    .output();
                gamma.true_mut().rewire_dependents(true_output, true_val);

                let false_output = gamma
                    .false_branch()
                    .to_node::<InputParam>(falsy_param)
                    .output();
                gamma.false_mut().rewire_dependents(false_output, false_val);
            }
        }

        changed |= true_visitor.visit_graph(gamma.true_mut());
        self.changes.combine(&true_visitor.changes);

        changed |= false_visitor.visit_graph(gamma.false_mut());
        self.changes.combine(&false_visitor.changes);

        for (&port, &param) in gamma.outputs().iter().zip(gamma.output_params()) {
            let true_output = gamma.true_branch().input_source(
                gamma
                    .true_branch()
                    .get_node(param[0])
                    .to_output_param()
                    .input(),
            );

            let false_output = gamma.false_branch().input_source(
                gamma
                    .false_branch()
                    .get_node(param[1])
                    .to_output_param()
                    .input(),
            );

            if let (Some(truthy), Some(falsy)) = (
                true_visitor.values.get(true_output),
                false_visitor.values.get(false_output),
            ) {
                if truthy.values_eq(falsy) {
                    tracing::trace!("propagating {:?} out of gamma node", truthy);
                    self.values.add(port, truthy);

                    let value = graph.constant(truthy);
                    graph.rewire_dependents(port, value.value());
                    self.changes.inc::<"const-gamma-output">();
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
        }
    }
}

// TODO: Make sure that constants are propagated into gammas and thetas
//       as well as out of gammas
test_opts! {
    constant_add,
    passes = |tape_len| -> Vec<Box<dyn Pass + 'static>> {
        bvec![ConstFolding::new(tape_len), Dce::new()]
    },
    output = [30],
    |graph, effect, tape_len| {
        let lhs = graph.int(Ptr::new(10, tape_len));
        let rhs = graph.int(Ptr::new(20, tape_len));
        let sum = graph.add(lhs.value(), rhs.value());

        graph.output(sum.value(), effect).output_effect()
    },
}

#[test]
fn const_add() {
    let mut input = {
        let mut graph = Rvsdg::new();

        let start = graph.start();
        let lhs = graph.byte(10);
        let rhs = graph.byte(20);
        let sum = graph.add(lhs.value(), rhs.value());
        let output = graph.output(sum.value(), start.effect());
        let _end = graph.end(output.output_effect());

        graph
    };
    crate::driver::run_opt_passes(
        &mut input,
        usize::MAX,
        &PassConfig::new(30_000, true, true),
        None,
    );

    let expected = {
        let mut graph = Rvsdg::new();

        let start = graph.start();
        let thirty = graph.byte(30);
        let output = graph.output(thirty.value(), start.effect());
        let _end = graph.end(output.output_effect());

        graph
    };

    assert!(input.structural_eq(&expected));
}

test_opts! {
    constant_mul,
    passes = |tape_len| -> Vec<Box<dyn Pass + 'static>> {
        bvec![ConstFolding::new(tape_len), Dce::new()]
    },
    output = [100],
    |graph, effect, tape_len| {
        let lhs = graph.int(Ptr::new(10, tape_len));
        let rhs = graph.int(Ptr::new(10, tape_len));
        let product = graph.mul(lhs.value(), rhs.value());

        graph.output(product.value(), effect).output_effect()
    },
}

test_opts! {
    constant_sub,
    passes = |tape_len| -> Vec<Box<dyn Pass + 'static>> {
        bvec![ConstFolding::new(tape_len), Dce::new()]
    },
    output = [245],
    |graph, effect, tape_len| {
        let lhs = graph.int(Ptr::new(10, tape_len));
        let rhs = graph.int(Ptr::new(20, tape_len));
        let difference = graph.sub(lhs.value(), rhs.value());

        graph.output(difference.value(), effect).output_effect()
    },
}

test_opts! {
    chained_booleans,
    passes = |tape_len| -> Vec<Box<dyn Pass + 'static>> {
        bvec![ConstFolding::new(tape_len), Dce::new()]
    },
    output = [1],
    |graph, effect, _| {
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
