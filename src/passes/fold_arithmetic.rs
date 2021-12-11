use crate::{
    graph::{
        Add, Bool, Gamma, InputParam, InputPort, Int, Neg, NodeExt, OutputPort, Rvsdg, Sub, Theta,
    },
    passes::{utils::ConstantStore, Pass},
};

/// Folds arithmetic operations together
pub struct FoldArithmetic {
    values: ConstantStore,
    changed: bool,
}

impl FoldArithmetic {
    pub fn new() -> Self {
        Self {
            values: ConstantStore::new(),
            changed: false,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    fn operand(&self, graph: &Rvsdg, input: InputPort) -> (OutputPort, Option<u32>) {
        let source = graph.input_source(input);
        let value = self.values.u32(source);

        (source, value)
    }
}

// TODO: Double bitwise and logical negation
impl Pass for FoldArithmetic {
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

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: u32) {
        self.values.add(int.value(), value);
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        self.values.add(bool.value(), value);
    }

    // TODO: `add x, neg y => sub x, y`, `add neg x, neg y => neg (add x, y)`
    fn visit_add(&mut self, graph: &mut Rvsdg, mut add: Add) {
        let ((lhs_src, lhs_val), (rhs_src, rhs_val)) = (
            self.operand(graph, add.lhs()),
            self.operand(graph, add.rhs()),
        );

        // If both operands are const we'll leave this to constant folding
        if lhs_val.is_some() && rhs_val.is_some() {
            return;
        }

        if let Some((lhs_sub, rhs_val)) = graph.cast_parent::<_, Sub>(lhs_src).copied().zip(rhs_val)
        {
            let (sub_rhs, sub_rhs_val) = self.operand(graph, lhs_sub.rhs());

            if let Some(sub_rhs_val) = sub_rhs_val {
                if sub_rhs_val > rhs_val {
                    let replacement = Sub::new(add.node(), add.lhs(), add.rhs(), add.value());
                    graph.replace_node(add.node(), replacement);

                    graph.remove_inputs(replacement.node());
                    graph.add_value_edge(graph.input_source(lhs_sub.lhs()), add.lhs());

                    let difference = graph.int(sub_rhs_val - rhs_val);
                    graph.add_value_edge(difference.value(), add.rhs());

                    self.changed();
                } else {
                    graph.remove_inputs(add.node());
                    graph.add_value_edge(graph.input_source(lhs_sub.lhs()), add.lhs());

                    let difference = graph.int(rhs_val - sub_rhs_val);
                    graph.add_value_edge(difference.value(), add.rhs());

                    self.changed();
                }
            }
        }
    }

    fn visit_sub(&mut self, graph: &mut Rvsdg, sub: Sub) {
        let ((lhs_src, lhs_val), (rhs_src, rhs_val)) = (
            self.operand(graph, sub.lhs()),
            self.operand(graph, sub.rhs()),
        );

        if let Some((lhs_add, rhs_val)) = graph.cast_parent::<_, Add>(lhs_src).copied().zip(rhs_val)
        {
            let (add_rhs, add_rhs_val) = self.operand(graph, lhs_add.rhs());

            if let Some(add_rhs_val) = add_rhs_val {
                if add_rhs_val > rhs_val {
                    let replacement = Add::new(sub.node(), sub.lhs(), sub.rhs(), sub.value());
                    graph.replace_node(sub.node(), replacement);

                    graph.remove_inputs(sub.node());
                    graph.add_value_edge(graph.input_source(lhs_add.lhs()), sub.lhs());

                    let difference = graph.int(add_rhs_val - rhs_val);
                    graph.add_value_edge(difference.value(), sub.rhs());

                    self.changed();
                } else {
                    graph.remove_inputs(sub.node());
                    graph.add_value_edge(graph.input_source(lhs_add.lhs()), sub.lhs());

                    let difference = graph.int(rhs_val - add_rhs_val);
                    graph.add_value_edge(difference.value(), sub.rhs());

                    self.changed();
                }
            }
        }
    }

    fn visit_neg(&mut self, graph: &mut Rvsdg, neg: Neg) {}

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut changed = false;
        let mut visitor = Self::new();

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

        if changed {
            graph.replace_node(theta.node(), theta);
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

            if let Some(constant) = self.values.get(output) {
                let true_param = gamma.true_branch().to_node::<InputParam>(truthy_param);
                truthy_visitor.values.add(true_param.output(), constant);

                let false_param = gamma.false_branch().to_node::<InputParam>(falsy_param);
                falsy_visitor.values.add(false_param.output(), constant);
            }
        }

        changed |= truthy_visitor.visit_graph(gamma.true_mut());
        changed |= falsy_visitor.visit_graph(gamma.false_mut());

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
            self.changed();
        }
    }
}

impl Default for FoldArithmetic {
    fn default() -> Self {
        Self::new()
    }
}
