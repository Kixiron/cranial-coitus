use crate::{
    graph::{
        Add, Bool, Gamma, InputParam, Int, Load, Node, NodeExt, OutputPort, Rvsdg, Start, Store,
        Theta,
    },
    passes::{utils::ConstantStore, Pass},
    utils::HashMap,
};

#[derive(Debug)]
pub struct ShiftCell {
    changed: bool,
    constants: ConstantStore,
    shifts_removed: usize,
}

impl ShiftCell {
    pub fn new() -> Self {
        Self {
            constants: ConstantStore::new(),
            changed: false,
            shifts_removed: 0,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    fn visit_gamma_theta(
        &mut self,
        graph: &mut Rvsdg,
        theta: &mut Theta,
    ) -> (bool, Option<(ShiftCandidate, Self)>) {
        let mut changed = false;
        let mut visitor = Self::new();

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        self.constants
            .theta_invariant_inputs_into(theta, graph, &mut visitor.constants);

        changed |= visitor.visit_graph(theta.body_mut());

        let candidate = self
            .theta_is_candidate(&visitor.constants, theta)
            .map(|candidate| (candidate, visitor));

        (changed, candidate)
    }

    fn inline_shift_gamma(
        &mut self,
        gamma: &Gamma,
        theta: &Theta,
        visitor: Self,
        graph: &mut Rvsdg,
        theta_graph: &Rvsdg,
        shift_candidate: ShiftCandidate,
    ) -> bool {
        let ShiftCandidate { src_ptr, dest_ptr } = shift_candidate;

        tracing::debug!(
            "found theta that composes to a shift loop copying from {:?} to {:?}",
            src_ptr,
            dest_ptr,
        );

        let mut get_gamma_input = |output| {
            visitor
                .constants
                .u32(output)
                .map(|int| graph.int(int).value())
                .or_else(|| {
                    theta
                        .invariant_input_pairs()
                        .find_map(|(port, input)| {
                            if input.output() == output {
                                Some(theta_graph.input_source(port))
                            } else {
                                None
                            }
                        })
                        .and_then(|output| {
                            self.constants
                                .u32(output)
                                .map(|int| graph.int(int).value())
                                .or_else(|| {
                                    gamma.inputs().iter().zip(gamma.input_params()).find_map(
                                        |(&port, &[_, input])| {
                                            let input =
                                                gamma.false_branch().to_node::<InputParam>(input);

                                            if input.output() == output {
                                                Some(graph.input_source(port))
                                            } else {
                                                None
                                            }
                                        },
                                    )
                                })
                        })
                })
        };

        if let (Some(src_ptr), Some(dest_ptr)) =
            (get_gamma_input(src_ptr), get_gamma_input(dest_ptr))
        {
            let input_effect = graph.input_source(gamma.effect_in());

            // Load the source value
            let src_val = graph.load(src_ptr, input_effect);

            // Store the source value into the destination cell
            let store_src_to_dest =
                graph.store(dest_ptr, src_val.output_value(), src_val.output_effect());

            // Unconditionally store 0 to the destination cell
            let zero = graph.int(0);
            let zero_dest_cell =
                graph.store(src_ptr, zero.value(), store_src_to_dest.output_effect());

            // Wire the final store into the gamma's output effect
            graph.rewire_dependents(gamma.effect_out(), zero_dest_cell.output_effect());

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
                                        (input.node() == param.node()).then_some(port)
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

            graph.remove_node(gamma.node());
            self.changed();

            true
        } else {
            tracing::trace!(
                "failed to optimize shift loop copying from {:?} to {:?}, \
                the pointers were not in the expected form yet",
                src_ptr,
                dest_ptr,
            );

            false
        }
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
    /// // TODO: Technically this should add src_val to dest_val
    /// store dest_ptr, src_val
    ///
    /// // Zero out the source cell
    /// store src_ptr, int 0
    /// ```
    fn theta_is_candidate(&self, values: &ConstantStore, theta: &Theta) -> Option<ShiftCandidate> {
        let graph = theta.body();

        // Get the body's start node
        let start = theta.start_node();

        // Get the first load, can either be `src_val := load src_ptr` or
        // `dest_val := load dest_ptr` depending on the configuration
        // of the shift loop
        let load_one = graph.cast_output_dest::<Load>(start.effect())?;
        let load_ptr_one = graph.input_source(load_one.ptr());

        // Get the first store, can either be `store src_ptr, src_dec` or
        // `store dest_ptr, dest_inc`
        let store_one = graph.cast_output_dest::<Store>(load_one.output_effect())?;
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
        let offset_one = values.u32(if lhs_operand == load_one.output_value() {
            rhs_operand
        } else if rhs_operand == load_one.output_value() {
            lhs_operand
        } else {
            return None;
        })?;

        // Get the second load
        let load_two = graph.cast_output_dest::<Load>(store_one.output_effect())?;
        let load_ptr_two = graph.input_source(load_two.ptr());

        // Get the second store
        let store_two = graph.cast_output_dest::<Store>(load_two.output_effect())?;
        let store_ptr_two = graph.input_source(store_two.ptr());

        // If the pointers aren't equal, bail
        if store_ptr_two != load_ptr_two {
            return None;
        }

        // Get the stored value, will either be `src_dec := add v60, int -1` or
        // `dest_inc := add dest_val, int 1`
        let add_two = graph.cast_input_source::<Add>(store_two.value())?;
        let (lhs_operand, rhs_operand) = (
            graph.input_source(add_two.lhs()),
            graph.input_source(add_two.rhs()),
        );

        // Get the offset being applied by figuring out which side is the loaded value
        let offset_two = values.u32(if lhs_operand == load_two.output_value() {
            rhs_operand
        } else if rhs_operand == load_two.output_value() {
            lhs_operand
        } else {
            return None;
        })?;

        // Make sure that the second store is the last effect in the body
        let end = theta.end_node();
        let store_two_effect_target = graph.output_dest(store_two.output_effect()).next()?;
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
        // FIXME: Depends on the relation between the add and sub node, not the operand
        let (src_ptr, src_dec, dest_ptr) = if offset_one == 1 && offset_two == 1 {
            (load_ptr_two, add_two.value(), load_ptr_one)
        } else if offset_one == 1 && offset_two == 1 {
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
        let (lhs_const, rhs_const) = (values.u32(lhs_operand), values.u32(rhs_operand));

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
        self.constants.clear();
        self.changed = false;
    }

    fn report(&self) -> HashMap<&'static str, usize> {
        map! {
            "shift loops" => self.shifts_removed,
        }
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        self.constants.add(bool.value(), value);
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: u32) {
        self.constants.add(int.value(), value);
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;

        let (mut true_visitor, mut false_visitor) = (Self::new(), Self::new());
        self.constants.gamma_inputs_into(
            &gamma,
            graph,
            &mut true_visitor.constants,
            &mut false_visitor.constants,
        );

        changed |= true_visitor.visit_graph(gamma.true_mut());
        changed |= false_visitor.visit_graph(gamma.false_mut());

        // Is `true` if the gamma's true branch is a passthrough
        let true_is_empty = {
            let start = gamma.true_branch().to_node::<Start>(gamma.starts()[0]);
            let next_node = gamma
                .true_branch()
                .get_output(start.effect())
                .unwrap()
                .0
                .node();

            next_node == gamma.ends()[0]
        };

        if true_is_empty {
            tracing::trace!(gamma = ?gamma.node(), "found gamma with empty true branch");
            let start = gamma.false_branch().to_node::<Start>(gamma.starts()[1]);

            // TODO: Make sure the next effect is the end node
            // TODO: Make sure the gamma's condition is correct
            if let Some(mut theta) = gamma
                .false_branch()
                .get_output(start.effect())
                .unwrap()
                .0
                .as_theta()
                .cloned()
            {
                tracing::trace!(
                    gamma = ?gamma.node(),
                    theta = ?theta.node(),
                    "found gamma with empty true branch and a false branch containing a theta",
                );

                let (gamma_changed, candidate) =
                    false_visitor.visit_gamma_theta(gamma.false_mut(), &mut theta);
                changed |= gamma_changed;

                if let Some((candidate, theta_body_visitor)) = candidate {
                    if self.inline_shift_gamma(
                        &gamma,
                        &theta,
                        theta_body_visitor,
                        graph,
                        gamma.false_branch(),
                        candidate,
                    ) {
                        tracing::trace!(
                            gamma = ?gamma.node(),
                            theta = ?theta.node(),
                            "inlined shift cell candidate",
                        );

                        self.shifts_removed += 1;
                        self.changed();

                        return;
                    }
                }
            }
        }

        if changed {
            graph.replace_node(gamma.node(), gamma);
            self.changed();
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut changed = false;

        let mut visitor = Self::new();
        self.constants
            .theta_invariant_inputs_into(&theta, graph, &mut visitor.constants);

        changed |= visitor.visit_graph(theta.body_mut());

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
