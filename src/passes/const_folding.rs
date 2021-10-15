use crate::{
    graph::{Add, Bool, EdgeKind, Eq, Int, NodeId, Not, Phi, Rvsdg, Theta},
    ir::Const,
    passes::Pass,
};
use std::collections::HashMap;

/// Evaluates constant operations within the program
pub struct ConstFolding {
    values: HashMap<NodeId, Const>,
    changed: bool,
}

impl ConstFolding {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            changed: false,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }
}

// TODO: Const fold `not`
impl Pass for ConstFolding {
    fn pass_name(&self) -> &str {
        "constant-folding"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    // TODO: Arithmetic folding
    // ```
    // _563 := add _562, _560
    // _564 := add _563, _560
    // _565 := add _564, _560
    // _566 := add _565, _560
    // _567 := add _566, _560
    // _568 := add _567, _560
    // _569 := add _568, _560
    // _570 := add _569, _560
    // ```
    fn visit_add(&mut self, graph: &mut Rvsdg, add: Add) {
        debug_assert_eq!(graph.incoming_count(add.node()), 2);

        let inputs @ [(_, lhs), (_, rhs)]: [_; 2] = graph
            .inputs(add.node())
            .map(|(input, operand, _, _)| {
                let value = operand.as_int().map(|(_, value)| value).or_else(|| {
                    self.values
                        .get(&operand.node_id())
                        .and_then(Const::convert_to_i32)
                });

                (input, value)
            })
            // TODO: Remove this alloc
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // If both sides of the add are known, we can evaluate it
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let sum = lhs + rhs;
            tracing::debug!("evaluated add {:?} to {}", add, sum);

            let int = graph.int(sum);
            self.values.insert(int.node(), sum.into());

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
        debug_assert_eq!(graph.incoming_count(eq.node()), 2);

        let [(lhs_node, lhs), (rhs_node, rhs)]: [_; 2] = graph
            .inputs(eq.node())
            .map(|(input, operand, _, _)| {
                let value = operand.as_int().map(|(_, value)| value).or_else(|| {
                    self.values
                        .get(&operand.node_id())
                        .and_then(Const::convert_to_i32)
                });

                (input, value)
            })
            // TODO: Remove this alloc
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

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
        debug_assert_eq!(graph.incoming_count(not.node()), 1);

        let (input_node, _, edge) = graph.get_input(not.input());
        debug_assert_eq!(edge, EdgeKind::Value);

        if let Some(value) = self
            .values
            .get(&input_node.node_id())
            .and_then(Const::as_bool)
        {
            tracing::debug!("constant folding 'not {}' to '{}'", value, !value);

            let inverted = graph.bool(!value);
            graph.rewire_dependents(not.value(), inverted.value());
            graph.remove_node(not.node());
        }
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        let replaced = self.values.insert(bool.node(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Bool(value)));
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: i32) {
        let replaced = self.values.insert(int.node(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Int(value)));
    }

    fn visit_phi(&mut self, graph: &mut Rvsdg, mut phi: Phi) {
        let (mut truthy_visitor, mut falsy_visitor) = (Self::new(), Self::new());

        // For each input into the phi region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[truthy_param, falsy_param]) in phi.inputs().iter().zip(phi.input_params()) {
            let (input_node, _, _) = graph.get_input(input);
            let input_node_id = input_node.node_id();

            if let Some(constant) = self.values.get(&input_node_id).cloned() {
                let replaced = truthy_visitor.values.insert(truthy_param, constant.clone());
                debug_assert!(replaced.is_none());

                let replaced = falsy_visitor.values.insert(falsy_param, constant);
                debug_assert!(replaced.is_none());
            }
        }

        // TODO: Eliminate phi branches based on phi condition

        truthy_visitor.visit_graph(phi.truthy_mut());
        falsy_visitor.visit_graph(phi.falsy_mut());
        self.changed |= truthy_visitor.did_change();
        self.changed |= falsy_visitor.did_change();

        // TODO: Propagate constants out of phi bodies?

        graph.replace_node(phi.node(), phi);
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut visitor = Self::new();

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &param) in theta.inputs().iter().zip(theta.input_params()) {
            let (input_node, _, _) = graph.get_input(input);
            let input_node_id = input_node.node_id();

            if let Some(constant) = self.values.get(&input_node_id).cloned() {
                let replaced = visitor.values.insert(param, constant);
                debug_assert!(replaced.is_none());
            }
        }

        visitor.visit_graph(theta.body_mut());
        self.changed |= visitor.did_change();

        // TODO: Propagate constants out of theta bodies?

        graph.replace_node(theta.node(), theta);
    }
}

impl Default for ConstFolding {
    fn default() -> Self {
        Self::new()
    }
}
