use crate::{
    graph::{Gamma, InputParam, Int, Node, NodeExt, OutputPort, Rvsdg, Theta},
    ir::Const,
    passes::Pass,
    utils::{AssertNone, HashMap},
};

pub struct ShiftCell {
    changed: bool,
    values: HashMap<OutputPort, Const>,
}

impl ShiftCell {
    pub fn new() -> Self {
        Self {
            values: HashMap::with_hasher(Default::default()),
            changed: false,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    /// The shift cell motif looks like this and is generated by the
    /// source program of `[>>+<<-]` where `>>` and `<<` can be any
    /// number of pointer increments or decrements
    ///
    /// ```
    /// // TODO: Use psi nodes to account for both orientations of this,
    /// //       increment-decrement and decrement-increment flavors
    /// do {
    ///     // Increment the destination cell
    ///     dest_val := load dest_ptr
    ///     dest_inc := add dest_val, int 1
    ///     store dest_ptr, dest_inc
    ///
    ///     // Decrement the source cell
    ///     src_val := load src_ptr
    ///     src_dec := add v60, int -1
    ///     store src_ptr, src_dec
    ///
    ///     // Test if the source cell is zero
    ///     src_eq_zero := eq src_dec, int 0
    ///     src_neq_zero := not src_eq_zero
    /// } while { src_neq_zero }
    /// ```
    ///
    /// After conversion, this should be the generated code:
    /// ```
    /// // Load the source value
    /// src_val := load src_ptr
    ///
    /// // Store the source value into the destination cell
    /// store dest_ptr, src_val
    ///
    /// // Zero out the source cell
    /// store src_ptr, int 0
    /// ```
    fn theta_is_candidate(
        &self,
        values: &HashMap<OutputPort, Const>,
        theta: &Theta,
    ) -> Option<ShiftCandidate> {
        let graph = theta.body();

        // Get the body's start node
        let start = theta.start_node();

        // Get the first load, can either be `src_val := load src_ptr` or
        // `dest_val := load dest_ptr` depending on the configuration
        // of the shift loop
        let load_one = graph.get_output(start.effect())?.0.as_load()?;
        let load_ptr_one = graph.input_source(load_one.ptr());

        // Get the first store, can either be `store src_ptr, src_dec` or
        // `store dest_ptr, dest_inc`
        let store_one = graph.get_output(load_one.effect())?.0.as_store()?;
        let store_ptr_one = graph.input_source(store_one.ptr());

        // If the pointers aren't equal, bail
        if store_ptr_one != load_ptr_one {
            return None;
        }

        // Get the stored value, will either be `src_dec := add v60, int -1` or
        // `dest_inc := add dest_val, int 1`
        let add_one = graph.input_source_node(store_one.value()).as_add()?;
        let (lhs_operand, rhs_operand) = (
            graph.input_source(add_one.lhs()),
            graph.input_source(add_one.rhs()),
        );

        // Get the offset being applied by figuring out which side is the loaded value
        let offset_one = values
            .get(&if lhs_operand == load_one.value() {
                rhs_operand
            } else if rhs_operand == load_one.value() {
                lhs_operand
            } else {
                return None;
            })?
            .as_int()?;

        // Get the second load
        let load_two = graph.get_output(store_one.effect())?.0.as_load()?;
        let load_ptr_two = graph.input_source(load_two.ptr());

        // Get the second store
        let store_two = graph.get_output(load_two.effect())?.0.as_store()?;
        let store_ptr_two = graph.input_source(store_two.ptr());

        // If the pointers aren't equal, bail
        if store_ptr_two != load_ptr_two {
            return None;
        }

        // Get the stored value, will either be `src_dec := add v60, int -1` or
        // `dest_inc := add dest_val, int 1`
        let add_two = graph.input_source_node(store_two.value()).as_add()?;
        let (lhs_operand, rhs_operand) = (
            graph.input_source(add_two.lhs()),
            graph.input_source(add_two.rhs()),
        );

        // Get the offset being applied by figuring out which side is the loaded value
        let offset_two = values
            .get(&if lhs_operand == load_two.value() {
                rhs_operand
            } else if rhs_operand == load_two.value() {
                lhs_operand
            } else {
                return None;
            })?
            .as_int()?;

        // Make sure that the second store is the last effect in the body
        let end = theta.end_node();
        let store_two_effect_target = graph.get_output(store_two.effect())?.1;
        if store_two_effect_target != end.input_effect() {
            return None;
        }

        // Ok, now that we have all of our relevant pointers and whatnot we can try
        // and figure out which orientation we're in, which is the destination and which
        // is the source pointer. We're following a relatively simple method:
        // - See which cell is being incremented by one
        // - See which cell is being decremented by one
        // - Make sure that the cell being decremented is also used
        //   in an `x != 0` condition for the enclosing theta
        // - If both cells are being incremented or decremented, this
        //   isn't the pattern we're looking for so we'll just bail

        // See which cell is being incremented and which is decremented
        let (src_ptr, src_dec, dest_ptr) = if offset_one == 1 && offset_two == -1 {
            (load_ptr_two, add_two.value(), load_ptr_one)
        } else if offset_one == -1 && offset_two == 1 {
            (load_ptr_one, add_one.value(), load_ptr_two)
        } else {
            return None;
        };

        // Should be `src_neq_zero := not src_eq_zero`
        let src_neq_zero = graph
            .input_source_node(theta.condition().input())
            .as_not()?;

        // Should be `src_eq_zero := eq src_dec, int 0`
        let src_eq_zero = graph.input_source_node(src_neq_zero.input()).as_eq()?;
        let (lhs_operand, rhs_operand) = (
            graph.input_source(src_eq_zero.lhs()),
            graph.input_source(src_eq_zero.rhs()),
        );
        let (lhs_const, rhs_const) = (
            values.get(&lhs_operand).and_then(Const::as_int),
            values.get(&rhs_operand).and_then(Const::as_int),
        );

        // Make sure that one of the operands of the eq is `src_dec` and the other is a zero
        if !((lhs_operand == src_dec && matches!(rhs_const, Some(0)))
            || (rhs_operand == src_dec && matches!(lhs_const, Some(0))))
        {
            return None;
        }

        // Yay, all of our validation is done and we've determined that this is
        // indeed a shift motif
        Some(ShiftCandidate::new(src_ptr, dest_ptr))
    }
}

impl Pass for ShiftCell {
    fn pass_name(&self) -> &str {
        "shift-cell"
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

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;
        let (mut truthy_visitor, mut falsy_visitor) = (Self::new(), Self::new());

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[true_param, false_param]) in gamma.inputs().iter().zip(gamma.input_params())
        {
            let (_, source, _) = graph.get_input(input);

            if let Some(constant) = self.values.get(&source).cloned() {
                let true_param = gamma.true_branch().to_node::<InputParam>(true_param);
                truthy_visitor
                    .values
                    .insert(true_param.output(), constant.clone())
                    .debug_unwrap_none();

                let false_param = gamma.false_branch().to_node::<InputParam>(false_param);
                falsy_visitor
                    .values
                    .insert(false_param.output(), constant)
                    .debug_unwrap_none();
            }
        }

        changed |= truthy_visitor.visit_graph(gamma.true_mut());
        changed |= falsy_visitor.visit_graph(gamma.false_mut());

        if changed {
            graph.replace_node(gamma.node(), gamma);
            self.changed();
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut changed = false;
        let mut visitor = Self::new();

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (input, param) in theta.input_pairs() {
            if let Some(constant) = self.values.get(&graph.input_source(input)).cloned() {
                visitor
                    .values
                    .insert(param.output(), constant)
                    .debug_unwrap_none();
            }
        }

        changed |= visitor.visit_graph(theta.body_mut());

        if let Some(ShiftCandidate { src_ptr, dest_ptr }) =
            self.theta_is_candidate(&visitor.values, &theta)
        {
            tracing::debug!(
                "found theta that composes to a shift loop copying from {:?} to {:?}",
                src_ptr,
                dest_ptr,
            );

            let mut get_theta_input = |output| {
                visitor
                    .values
                    .get(&output)
                    .and_then(Const::as_int)
                    .map(|int| graph.int(int).value())
                    .or_else(|| {
                        theta.invariant_input_pairs().find_map(|(port, input)| {
                            if input.output() == output {
                                Some(graph.get_input(port).1)
                            } else {
                                None
                            }
                        })
                    })
            };

            if let (Some(src_ptr), Some(dest_ptr)) =
                (get_theta_input(src_ptr), get_theta_input(dest_ptr))
            {
                let input_effect = graph.input_source(theta.input_effect().unwrap());

                // Load the source value
                let src_val = graph.load(src_ptr, input_effect);

                // Store the source value into the destination cell
                let store_src_to_dest = graph.store(dest_ptr, src_val.value(), src_val.effect());

                // Unconditionally store 0 to the destination cell
                let zero = graph.int(0);
                let zero_dest_cell = graph.store(src_ptr, zero.value(), store_src_to_dest.effect());

                // Wire the final store into the theta's output effect
                graph.rewire_dependents(theta.output_effect().unwrap(), zero_dest_cell.effect());

                for (port, param) in theta.output_pairs() {
                    if let Some((input_node, ..)) = theta.body().try_input(param.input()) {
                        match *input_node {
                            Node::Int(_, value) => {
                                let int = graph.int(value);
                                graph.rewire_dependents(port, int.value());
                            }

                            Node::Bool(_, value) => {
                                let bool = graph.bool(value);
                                graph.rewire_dependents(port, bool.value());
                            }

                            Node::InputParam(param) => {
                                let input_value = graph.input_source(
                                    theta
                                        .input_pairs()
                                        .find_map(|(port, input)| {
                                            (input.node() == param.node()).then(|| port)
                                        })
                                        .unwrap(),
                                );

                                graph.rewire_dependents(port, input_value);
                            }

                            ref other => {
                                tracing::error!("missed output value from theta {:?}", other);
                            }
                        }
                    } else {
                        tracing::error!(
                            "output value from theta had no inputs {:?}->{:?}",
                            param,
                            port,
                        );
                    }
                }

                graph.remove_node(theta.node());
                self.changed();

                return;
            } else {
                tracing::trace!(
                    "failed to optimize shift loop copying from {:?} to {:?}, \
                    the pointers were not in the expected form yet",
                    src_ptr,
                    dest_ptr,
                );
            }
        }

        if changed {
            graph.replace_node(theta.node(), theta);
            self.changed();
        }
    }
}

impl Default for ShiftCell {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
struct ShiftCandidate {
    src_ptr: OutputPort,
    dest_ptr: OutputPort,
}

impl ShiftCandidate {
    fn new(src_ptr: OutputPort, dest_ptr: OutputPort) -> Self {
        Self { src_ptr, dest_ptr }
    }
}
