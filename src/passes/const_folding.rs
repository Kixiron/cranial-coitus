use crate::{
    graph::{Add, Bool, EdgeKind, Eq, Gamma, InputPort, Int, Not, OutputPort, Rvsdg, Theta},
    ir::Const,
    passes::Pass,
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

    fn operand(&self, graph: &Rvsdg, input: InputPort) -> (InputPort, Option<i32>) {
        let (operand, output, _) = graph.get_input(input);
        let value = operand
            .as_int()
            .map(|(_, value)| value)
            .or_else(|| self.values.get(&output).and_then(Const::convert_to_i32));

        (input, value)
    }
}

// TODO: neg
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

    fn visit_add(&mut self, graph: &mut Rvsdg, add: Add) {
        // debug_assert_eq!(graph.incoming_count(add.node()), 2);

        let inputs @ [(_, lhs), (_, rhs)] = [
            self.operand(graph, add.lhs()),
            self.operand(graph, add.rhs()),
        ];

        // If both sides of the add are known, we can evaluate it
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let sum = lhs + rhs;
            tracing::debug!("evaluated add {:?} to {}", add, sum);

            let int = graph.int(sum);
            self.values.insert(int.value(), sum.into());

            graph.rewire_dependents(add.value(), int.value());
            graph.remove_node(add.node());

            self.changed();

        // If either side of the add is zero, we can remove the add entirely
        } else if let [(_, Some(0)), (input, None)] | [(input, None), (_, Some(0))] = inputs {
            let non_zero_value = graph.input_source(input);
            tracing::debug!(
                "removing an addition by zero {:?} into a direct value of {:?}",
                add,
                non_zero_value,
            );

            graph.rewire_dependents(add.value(), non_zero_value);
            graph.remove_node(add.node());

            self.changed();
        }
    }

    fn visit_eq(&mut self, graph: &mut Rvsdg, eq: Eq) {
        // debug_assert_eq!(graph.incoming_count(eq.node()), 2);

        let [(lhs_node, lhs), (rhs_node, rhs)] =
            [self.operand(graph, eq.lhs()), self.operand(graph, eq.rhs())];

        // If both values are known we can statically evaluate the comparison
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            tracing::debug!(
                "replaced const eq with {} ({:?} == {:?}) {:?} ({:?} == {:?})",
                lhs == rhs,
                lhs,
                rhs,
                eq,
                graph.get_input(lhs_node),
                graph.get_input(rhs_node),
            );

            let true_val = graph.bool(lhs == rhs);
            graph.rewire_dependents(eq.value(), true_val.value());
            graph.remove_node(eq.node());

            self.changed();

        // If the operands are equal this comparison will always be true
        } else if graph.get_input(lhs_node).0.node_id() == graph.get_input(rhs_node).0.node_id() {
            tracing::debug!(
                "replaced self-equality with true ({:?} == {:?}) {:?}",
                lhs_node,
                rhs_node,
                eq
            );

            let true_val = graph.bool(true);
            graph.rewire_dependents(eq.value(), true_val.value());
            graph.remove_node(eq.node());

            self.changed();
        }
    }

    fn visit_not(&mut self, graph: &mut Rvsdg, not: Not) {
        // debug_assert_eq!(graph.incoming_count(not.node()), 1);

        let (_, output, edge) = graph.get_input(not.input());
        debug_assert_eq!(edge, EdgeKind::Value);

        if let Some(value) = self.values.get(&output).and_then(Const::as_bool) {
            tracing::debug!("constant folding 'not {}' to '{}'", value, !value);

            let inverted = graph.bool(!value);
            graph.rewire_dependents(not.value(), inverted.value());
            graph.remove_node(not.node());

            self.changed();
        }
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        let replaced = self.values.insert(bool.value(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Bool(value)));
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: i32) {
        let replaced = self.values.insert(int.value(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Int(value)));
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let (mut truthy_visitor, mut falsy_visitor) = (Self::new(), Self::new());

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[truthy_param, falsy_param]) in
            gamma.inputs().iter().zip(gamma.input_params())
        {
            let (_, output, _) = graph.get_input(input);

            if let Some(constant) = self.values.get(&output).cloned() {
                let true_param = gamma.true_branch().get_node(truthy_param).to_input_param();
                let replaced = truthy_visitor
                    .values
                    .insert(true_param.value(), constant.clone());
                debug_assert!(replaced.is_none());

                let false_param = gamma.false_branch().get_node(falsy_param).to_input_param();
                let replaced = falsy_visitor.values.insert(false_param.value(), constant);
                debug_assert!(replaced.is_none());
            }
        }

        // TODO: Eliminate gamma branches based on gamma condition

        truthy_visitor.visit_graph(gamma.truthy_mut());
        falsy_visitor.visit_graph(gamma.falsy_mut());
        self.changed |= truthy_visitor.did_change();
        self.changed |= falsy_visitor.did_change();

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

        graph.replace_node(gamma.node(), gamma);
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut visitor = Self::new();

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &param) in theta.inputs().iter().zip(theta.input_params()) {
            let (_, output, _) = graph.get_input(input);

            if let Some(constant) = self.values.get(&output).cloned() {
                let replaced = visitor.values.insert(
                    theta.body().get_node(param).to_input_param().value(),
                    constant,
                );
                debug_assert!(replaced.is_none());
            }
        }

        visitor.visit_graph(theta.body_mut());
        self.changed |= visitor.did_change();

        for (&port, &param) in theta.outputs().iter().zip(theta.output_params()) {
            let param_input = theta.body().get_node(param).to_output_param().input();

            if let Some(value) = self
                .values
                .get(&theta.body().get_input(param_input).1)
                .cloned()
            {
                tracing::trace!("propagating {:?} out of theta node", value);
                self.values.insert(port, value);
            }
        }

        graph.replace_node(theta.node(), theta);
    }
}

impl Default for ConstFolding {
    fn default() -> Self {
        Self::new()
    }
}
