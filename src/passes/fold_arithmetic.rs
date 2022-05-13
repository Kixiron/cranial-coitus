use crate::{
    graph::{
        Add, Bool, Byte, Gamma, InputParam, InputPort, Int, Neg, NodeExt, Not, OutputPort, Rvsdg,
        Sub, Theta,
    },
    passes::{
        utils::{ChangeReport, Changes, ConstantStore},
        Pass,
    },
    values::{Cell, Ptr},
};

/// Folds arithmetic operations together
pub struct FoldArithmetic {
    values: ConstantStore,
    changes: Changes<5>,
    tape_len: u16,
}

impl FoldArithmetic {
    pub fn new(tape_len: u16) -> Self {
        Self {
            values: ConstantStore::new(tape_len),
            changes: Changes::new([
                "add-sub-greater",
                "add-sub-less-eq",
                "sub-add-greater",
                "sub-add-less-eq",
                "sub-sub",
            ]),
            tape_len,
        }
    }

    fn operand(&self, graph: &Rvsdg, input: InputPort) -> (OutputPort, Option<Ptr>) {
        let source = graph.input_source(input);
        let value = self.values.ptr(source);

        (source, value)
    }
}

// TODO: Double bitwise and logical negation
impl Pass for FoldArithmetic {
    fn pass_name(&self) -> &'static str {
        "fold-arithmetic"
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

    // TODO: `add x, neg y => sub x, y`, `add neg x, neg y => neg (add x, y)`
    fn visit_add(&mut self, graph: &mut Rvsdg, add: Add) {
        let ((lhs_src, lhs_val), (_, rhs_val)) = (
            self.operand(graph, add.lhs()),
            self.operand(graph, add.rhs()),
        );

        // If both operands are const we'll leave this to constant folding
        if lhs_val.is_some() && rhs_val.is_some() {
            return;
        }

        if let Some((lhs_sub, rhs_val)) = graph.cast_parent::<_, Sub>(lhs_src).copied().zip(rhs_val)
        {
            let (_, sub_rhs_val) = self.operand(graph, lhs_sub.rhs());

            if let Some(sub_rhs_val) = sub_rhs_val {
                let lhs_source = graph.input_source(lhs_sub.lhs());

                // `add (sub x, y), z where y > z => sub x, (y - z)`
                if sub_rhs_val > rhs_val {
                    let replacement = Sub::new(add.node(), add.lhs(), add.rhs(), add.value());
                    graph.replace_node(add.node(), replacement);

                    graph.remove_inputs(replacement.node());
                    graph.add_value_edge(lhs_source, add.lhs());

                    let difference = graph.int(sub_rhs_val - rhs_val);
                    graph.add_value_edge(difference.value(), add.rhs());

                    tracing::debug!(
                        add = ?add.node(),
                        "turned add (sub {sub_lhs}, {sub_rhs}), {add_rhs} where {sub_rhs} > {add_rhs} \
                        into sub {sub_lhs}, ({sub_rhs} - {add_rhs} = {diff})",
                        sub_lhs = lhs_source,
                        sub_rhs = sub_rhs_val,
                        add_rhs = rhs_val,
                        diff = sub_rhs_val - rhs_val,
                    );

                    self.changes.inc::<"add-sub-greater">();

                // `add (sub x, y), z where y <= z => add x, (z - y)`
                } else {
                    graph.remove_inputs(add.node());
                    graph.add_value_edge(lhs_source, add.lhs());

                    let difference = graph.int(rhs_val - sub_rhs_val);
                    graph.add_value_edge(difference.value(), add.rhs());

                    tracing::debug!(
                        add = ?add.node(),
                        "turned add (sub {sub_lhs}, {sub_rhs}), {add_rhs} where {sub_rhs} <= {add_rhs} \
                        into add {sub_lhs}, ({add_rhs} - {sub_rhs} = {diff})",
                        sub_lhs = lhs_source,
                        sub_rhs = sub_rhs_val,
                        add_rhs = rhs_val,
                        diff = rhs_val - sub_rhs_val,
                    );

                    self.changes.inc::<"add-sub-less-eq">();
                }
            }
        }
    }

    fn visit_sub(&mut self, graph: &mut Rvsdg, sub: Sub) {
        let ((lhs_src, _), (_, rhs_val)) = (
            self.operand(graph, sub.lhs()),
            self.operand(graph, sub.rhs()),
        );

        if let Some((lhs_add, rhs_val)) = graph.cast_parent::<_, Add>(lhs_src).copied().zip(rhs_val)
        {
            let (_, add_rhs_val) = self.operand(graph, lhs_add.rhs());

            if let Some(add_rhs_val) = add_rhs_val {
                let lhs_source = graph.input_source(lhs_add.lhs());

                // `sub (add x, y), z where y > z => add x, (y - z)`
                if add_rhs_val > rhs_val {
                    let replacement = Add::new(sub.node(), sub.lhs(), sub.rhs(), sub.value());
                    graph.replace_node(sub.node(), replacement);

                    graph.remove_inputs(sub.node());
                    graph.add_value_edge(lhs_source, sub.lhs());

                    let difference = graph.int(add_rhs_val - rhs_val);
                    graph.add_value_edge(difference.value(), sub.rhs());

                    tracing::debug!(
                        sub = ?sub.node(),
                        "turned sub (add {add_lhs}, {add_rhs}), {sub_rhs} where {add_rhs} > {sub_rhs} \
                        into add {add_lhs}, ({add_rhs} - {sub_rhs} = {diff})",
                        add_lhs = lhs_source,
                        add_rhs = add_rhs_val,
                        sub_rhs = rhs_val,
                        diff = add_rhs_val - rhs_val,
                    );

                    self.changes.inc::<"sub-add-greater">();

                // `sub (add x, y), z where y <= z => sub x, (z - y)`
                } else {
                    graph.remove_inputs(sub.node());
                    graph.add_value_edge(graph.input_source(lhs_add.lhs()), sub.lhs());

                    let difference = graph.int(rhs_val - add_rhs_val);
                    graph.add_value_edge(difference.value(), sub.rhs());

                    tracing::debug!(
                        sub = ?sub.node(),
                        "turned sub (add {add_lhs}, {add_rhs}), {sub_rhs} where {add_rhs} <= {sub_rhs} \
                        into sub {add_lhs}, ({sub_rhs} - {add_rhs} = {diff})",
                        add_lhs = lhs_source,
                        add_rhs = add_rhs_val,
                        sub_rhs = rhs_val,
                        diff = rhs_val - add_rhs_val,
                    );

                    self.changes.inc::<"sub-add-less-eq">();
                }
            }
        } else {
            // (x - const) - const ≡ x - (const + const)
            // (x - 1) - 1 ≡ x - (1 + 1) ≡ x - 2
            let _: Option<()> = try {
                let first_rhs = self.values.get(graph.input_source(sub.rhs()))?;

                let lhs_sub = *graph.cast_input_source::<Sub>(sub.lhs())?;
                let second_rhs = self.values.get(graph.input_source(lhs_sub.rhs()))?;

                let sum = first_rhs + second_rhs;
                let new_rhs = graph.constant(sum);
                self.values.add(new_rhs.value(), sum);

                graph.remove_inputs(sub.node());

                let value = graph.input_source(lhs_sub.lhs());
                graph.add_value_edge(value, sub.lhs());
                graph.add_value_edge(new_rhs.value(), sub.rhs());

                self.changes.inc::<"sub-sub">();
            };
        }
    }

    // TODO: Neg and not
    fn visit_neg(&mut self, _graph: &mut Rvsdg, _neg: Neg) {}
    fn visit_not(&mut self, _graph: &mut Rvsdg, _not: Not) {}

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut changed = false;
        let mut visitor = Self::new(self.tape_len);

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        // Note: We only propagate **invariant** inputs into the loop, propagating
        //       variant inputs requires dataflow information
        for (input, param) in theta.invariant_input_pairs() {
            if let Some(constant) = self.values.get(graph.input_source(input)) {
                visitor.values.add(param.output(), constant);
            }
        }

        changed |= visitor.visit_graph(theta.body_mut());
        self.changes.combine(&visitor.changes);

        if changed {
            graph.replace_node(theta.node(), theta);
        }
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;
        let (mut truthy_visitor, mut falsy_visitor) =
            (Self::new(self.tape_len), Self::new(self.tape_len));

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[truthy_param, falsy_param]) in
            gamma.inputs().iter().zip(gamma.input_params())
        {
            let (_, output, _) = graph.get_input(input);

            if let Some(constant) = self.values.get(output) {
                let true_param = gamma.true_branch().to_node::<InputParam>(truthy_param);
                truthy_visitor.values.add(true_param.output(), constant);

                let false_param = gamma.false_branch().to_node::<InputParam>(falsy_param);
                falsy_visitor.values.add(false_param.output(), constant);
            }
        }

        changed |= truthy_visitor.visit_graph(gamma.true_mut());
        self.changes.combine(&truthy_visitor.changes);

        changed |= falsy_visitor.visit_graph(gamma.false_mut());
        self.changes.combine(&falsy_visitor.changes);

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
                truthy_visitor.values.get(true_output),
                falsy_visitor.values.get(false_output),
            ) {
                if truthy == falsy {
                    tracing::trace!("propagating {:?} out of gamma node", truthy);
                    self.values.add(port, truthy);
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
