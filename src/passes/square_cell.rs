use crate::{
    graph::{
        Add, End, Eq, Gamma, InputParam, InputPort, Int, Load, NodeExt, NodeId, Not, OutputPort,
        Rvsdg, Start, Store, Theta,
    },
    ir::Const,
    passes::Pass,
    utils::{AssertNone, HashMap},
};
use std::collections::BTreeMap;

/// Turn the multiplication of a cell by itself (x²) into a multiplication instruction
///
/// ```
/// // Preconditions:
/// // - `src_ptr` and `temp1_ptr` should point to cells with values of zero
/// // - `temp0_ptr` should point to a cell with an integer value
///
/// if src_is_zero {
///     // `0 × 0 ≡ 0`, do nothing
/// } else {
///     // Otherwise we're multiplying a non-zero integer, `x × x ≡ x²`
///
///     do {
///         // Subtract 1 from temp0
///         temp0_val := load temp0_ptr
///         temp0_minus_one := add temp0_val, int -1
///         store temp0_ptr, temp0_minus_one
///
///         // Check if temp0 is zero
///         temp0_eq_zero := eq temp0_minus_one, int 0
///         if temp0_eq_zero {
///             // If temp0 is zero, pass
///         } else {
///             // Otherwise, perform the inner loop
///             // TODO: Use psi nodes to account for different configurations of the
///             //       various loads & stores
///             do {
///                 // Add 1 to temp1
///                 temp1_val := load temp1_ptr
///                 temp1_plus_one := add temp1_val, int 1
///                 store temp1_ptr, temp1_plus_one
///
///                 // Add 2 (??) to src
///                 src_val := load src_ptr
///                 // TODO: Will this always be two?
///                 src_plus_two := add src_val, int 2
///                 store src_ptr, src_plus_two
///
///                 // Subtract 1 from temp0
///                 temp0_val := load temp0_ptr
///                 temp0_minus_one := add temp0_val, int -1
///                 store temp0_ptr, temp0_minus_one
///
///                 // Keep looping while temp0 is non-zero
///                 temp0_eq_zero := eq temp0_minus_one, int 0
///                 temp0_not_zero := not temp0_eq_zero
///             } while { temp0_not_zero }
///         }
///
///         // TODO: Use psi nodes to account for different configurations of the
///         //       src & temp1 loads/stores
///
///         // Add 1 to src
///         src_val := load src_ptr
///         src_plus_one := add src_val, int 1
///         store src_ptr, src_plus_one
///
///         // Copy temp1's value to temp0
///         temp1_val := load temp1_ptr
///         store temp0_ptr, temp1_val
///
///         // Zero out temp1
///         store temp1_ptr, int 0
///
///         // Keep looping while temp1's old value (now temp0's value) is non-zero
///         temp1_eq_zero := eq temp1_val, int 0
///         temp1_not_zero := not temp1_eq_zero
///     } while { temp1_not_zero }
/// }
///
/// // Postconditions:
/// // - `src_ptr` points to a cell with the value of `(*src_ptr)²`
/// // - `temp0_ptr` and `temp1_ptr` both point to cells with values of zero
/// ```
///
/// After conversion, this should be the generated code:
/// ```
/// // Load the source value
/// src_val := load src_ptr
///
/// // Multiply the source value by itself
/// src_squared := mul src_val, src_val
///
/// // Store the squared value into the source cell
/// store src_ptr, src_squared
///
/// // Zero out temp0 and temp1's cells
/// store temp0_ptr, int 0
/// store temp1_ptr, int 0
/// ```
#[derive(Debug)]
pub struct SquareCell {
    changed: bool,
    values: BTreeMap<OutputPort, Const>,
    squares_removed: usize,
}

impl SquareCell {
    pub fn new() -> Self {
        Self {
            values: BTreeMap::new(),
            changed: false,
            squares_removed: 0,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    /// Returns the [`NodeId`] of the inner theta node within the gamma node if it exists
    /// and the pattern is matched
    ///
    /// Note that the innards of the do-while loop are not checked for their candidacy
    ///
    /// ```
    /// // Preconditions:
    /// // - `src_ptr` and `temp1_ptr` should point to cells with values of zero
    /// // - `temp0_ptr` should point to a cell with an integer value
    ///
    /// if src_is_zero {
    /// } else {
    ///     do { ??? } while { ??? }
    /// }
    /// ```
    ///
    fn outer_gamma_is_candidate(&self, gamma: &Gamma) -> Option<NodeId> {
        // TODO: Check preconditions
        // TODO: Check the gamma's condition

        // Make sure that the true branch is empty
        if gamma_branch_is_empty(gamma.true_branch()) {
            let graph = gamma.false_branch();

            let start = graph.to_node::<Start>(graph.start_nodes()[0]);
            let end = graph.to_node::<End>(graph.end_nodes()[0]);

            // Make sure the next effect after the start node is a theta node
            if let Some(theta) = graph.cast_output_dest::<Theta>(start.effect()) {
                // Make sure the theta is effectual (should be infallible due to it having an input effect)
                if let Some(output_effect) = theta.output_effect() {
                    // Return whether or not the theta's effect directly links to the branch's end node
                    if output_effect == graph.input_source(end.input_effect()) {
                        return Some(theta.node());
                    }
                }
            }
        }

        None
    }

    /// Check that the outmost theta within the outer gamma's false branch is in the proper form
    ///
    /// Note that the innards of the inner gamma node are not checked for their candidacy
    ///
    /// ```
    /// do {
    ///     // Subtract 1 from temp0
    ///     temp0_val := load temp0_ptr
    ///     temp0_minus_one := add temp0_val, int -1
    ///     store temp0_ptr, temp0_minus_one
    ///
    ///     // Check if temp0 is zero
    ///     temp0_eq_zero := eq temp0_minus_one, int 0
    ///     if temp0_eq_zero { ... } else { ... }
    ///
    ///     // TODO: Use psi nodes to account for different configurations of the
    ///     //       src & temp1 loads/stores
    ///
    ///     // Add 1 to src
    ///     src_val := load src_ptr
    ///     src_plus_one := add src_val, int 1
    ///     store src_ptr, src_plus_one
    ///
    ///     // Copy temp1's value to temp0
    ///     temp1_val := load temp1_ptr
    ///     store temp0_ptr, temp1_val
    ///
    ///     // Zero out temp1
    ///     store temp1_ptr, int 0
    ///
    ///     // Keep looping while temp1's old value (now temp0's value) is non-zero
    ///     temp1_eq_zero := eq temp1_val, int 0
    ///     temp1_not_zero := not temp1_eq_zero
    /// } while { temp1_not_zero }
    /// ```
    ///
    fn outmost_theta_is_candidate(
        &self,
        values: &BTreeMap<OutputPort, Const>,
        theta: &Theta,
    ) -> Option<OutmostTheta> {
        let graph = theta.body();

        let start = theta.start_node();

        // `temp0_val := load temp0_ptr`
        let temp0_load = graph.cast_output_dest::<Load>(start.effect())?;
        let temp0_ptr = graph.input_source(temp0_load.ptr());

        // `store temp0_ptr, temp0_minus_one`
        let temp0_store = graph.cast_output_dest::<Store>(temp0_load.output_effect())?;
        if graph.input_source(temp0_store.ptr()) != temp0_ptr {
            return None;
        }

        // `temp0_minus_one := add temp0_val, int -1`
        let temp0_minus_one = graph.cast_input_source::<Add>(temp0_store.value())?;
        if !ports_match(
            graph,
            values,
            (temp0_minus_one.lhs(), temp0_minus_one.rhs()),
            temp0_load.output_value(),
            -1,
        ) {
            return None;
        }

        // `if temp0_eq_zero { ... } else { ... }`
        let inner_gamma = graph.cast_output_dest::<Gamma>(temp0_store.output_effect())?;

        // `temp0_eq_zero := eq temp0_minus_one, int 0`
        let temp0_eq_zero = graph.cast_input_source::<Eq>(inner_gamma.condition())?;
        if !ports_match(
            graph,
            values,
            (temp0_eq_zero.lhs(), temp0_eq_zero.rhs()),
            temp0_minus_one.value(),
            0,
        ) {
            return None;
        }

        // `src_val := load src_ptr`
        let src_load = graph.cast_output_dest::<Load>(inner_gamma.effect_out())?;
        let src_ptr = graph.input_source(src_load.ptr());

        if src_ptr == temp0_ptr {
            return None;
        }

        // `store src_ptr, src_plus_one`
        let src_store = graph.cast_output_dest::<Store>(src_load.output_effect())?;
        if graph.input_source(src_store.ptr()) != src_ptr {
            return None;
        }

        // `src_plus_one := add src_val, int 1`
        let src_plus_one = graph.cast_input_source::<Add>(src_store.value())?;
        if !ports_match(
            graph,
            values,
            (src_plus_one.lhs(), src_plus_one.rhs()),
            src_load.output_value(),
            1,
        ) {
            return None;
        }

        // `temp1_val := load temp1_ptr`
        let temp1_load = graph.cast_output_dest::<Load>(src_store.output_effect())?;
        let temp1_ptr = graph.input_source(temp1_load.ptr());

        if temp1_ptr == src_ptr || temp1_ptr == temp0_ptr {
            return None;
        }

        // `store temp0_ptr, temp1_val`
        let temp0_store_with_temp1_val =
            graph.cast_output_dest::<Store>(temp1_load.output_effect())?;
        if graph.input_source(temp0_store_with_temp1_val.ptr()) != temp0_ptr
            || graph.input_source(temp0_store_with_temp1_val.value()) != temp1_load.output_value()
        {
            return None;
        }

        // `store temp1_ptr, int 0`
        let zero_temp1 =
            graph.cast_output_dest::<Store>(temp0_store_with_temp1_val.output_effect())?;
        if graph.input_source(zero_temp1.ptr()) != temp1_ptr
            || values
                .get(&graph.input_source(zero_temp1.value()))
                .and_then(Const::convert_to_i32)
                != Some(0)
        {
            return None;
        }

        // `temp1_not_zero := not temp1_eq_zero`
        let temp1_not_zero = graph.cast_input_source::<Not>(theta.condition().input())?;

        // `temp1_eq_zero := eq temp1_val, int 0`
        let temp1_eq_zero = graph.cast_input_source::<Eq>(temp1_not_zero.input())?;
        if !ports_match(
            graph,
            values,
            (temp1_eq_zero.lhs(), temp0_eq_zero.rhs()),
            temp1_load.output_value(),
            0,
        ) {
            return None;
        }

        let candidate = OutmostTheta::new(inner_gamma.node(), src_ptr, temp0_ptr, temp1_ptr);
        Some(candidate)
    }

    /// ```
    /// if temp0_eq_zero {
    ///     // If temp0 is zero, pass
    /// } else {
    ///     // Otherwise, perform the inner loop
    ///     do { ... } while { ... }
    /// }
    /// ```
    fn inner_gamma_is_candidate(&self, gamma: &Gamma) -> Option<InnerGamma> {
        // Make sure the true branch is empty
        {
            let graph = gamma.true_branch();
            debug_assert_eq!(graph.start_nodes().len(), 1);
            debug_assert_eq!(graph.end_nodes().len(), 1);

            let start = graph.to_node::<Start>(graph.start_nodes()[0]);

            if gamma.true_branch().output_dest_id(start.effect())? != graph.end_nodes()[0] {
                return None;
            }
        }

        let graph = gamma.false_branch();
        debug_assert_eq!(graph.start_nodes().len(), 1);
        debug_assert_eq!(graph.end_nodes().len(), 1);

        let start = graph.to_node::<Start>(graph.start_nodes()[0]);

        // `do { ... } while { ... }`
        let theta = graph.cast_output_dest::<Theta>(start.effect())?;
        if graph.output_dest_id(theta.output_effect()?)? != graph.end_nodes()[0] {
            return None;
        }

        Some(InnerGamma::new(theta.node()))
    }

    /// ```
    /// // Add 1 to temp1
    /// temp1_val := load temp1_ptr
    /// temp1_plus_one := add temp1_val, int 1
    /// store temp1_ptr, temp1_plus_one
    ///
    /// // Add 2 (??) to src
    /// src_val := load src_ptr
    /// // TODO: Will this always be two?
    /// src_plus_two := add src_val, int 2
    /// store src_ptr, src_plus_two
    ///
    /// // Subtract 1 from temp0
    /// temp0_val := load temp0_ptr
    /// temp0_minus_one := add temp0_val, int -1
    /// store temp0_ptr, temp0_minus_one
    ///
    /// // Keep looping while temp0 is non-zero
    /// temp0_eq_zero := eq temp0_minus_one, int 0
    /// temp0_not_zero := not temp0_eq_zero
    /// ```
    fn inner_theta_is_candidate(
        &self,
        theta: &Theta,
        values: &BTreeMap<OutputPort, Const>,
    ) -> Option<()> {
        let graph = theta.body();
        let start = theta.start_node();

        // `temp1_val := load temp1_ptr`
        let temp1_load = graph.cast_output_dest::<Load>(start.effect())?;
        let temp1_ptr = graph.input_source(temp1_load.ptr());

        // `store temp1_ptr, temp1_plus_one`
        let temp1_store = graph.cast_output_dest::<Store>(temp1_load.output_effect())?;
        if graph.input_source(temp1_store.ptr()) != temp1_ptr {
            return None;
        }

        // `temp1_plus_one := add temp1_val, int 1`
        let temp1_plus_one = graph.cast_input_source::<Add>(temp1_store.value())?;
        if !ports_match(
            graph,
            values,
            (temp1_plus_one.lhs(), temp1_plus_one.rhs()),
            temp1_load.output_value(),
            1,
        ) {
            return None;
        }

        // `src_val := load src_ptr`
        let src_load = graph.cast_output_dest::<Load>(temp1_store.output_effect())?;
        let src_ptr = graph.input_source(src_load.ptr());

        // `store src_ptr, src_plus_two`
        let src_store = graph.cast_output_dest::<Store>(src_load.output_effect())?;
        if graph.input_source(src_store.ptr()) != src_ptr {
            return None;
        }

        // `src_plus_two := add src_val, int 2`
        let src_plus_two = graph.cast_input_source::<Add>(src_store.value())?;
        if !ports_match(
            graph,
            values,
            (src_plus_two.lhs(), src_plus_two.rhs()),
            src_load.output_value(),
            2,
        ) {
            return None;
        }

        // `temp0_val := load temp0_ptr`
        let temp0_load = graph.cast_output_dest::<Load>(src_store.output_effect())?;
        let temp0_ptr = graph.input_source(temp0_load.ptr());

        // `store temp0_ptr, temp0_minus_one`
        let temp0_store = graph.cast_output_dest::<Store>(temp0_load.output_effect())?;
        if graph.input_source(temp0_store.ptr()) != temp0_ptr {
            return None;
        }

        // `temp0_minus_one := add temp0_val, int -1`
        let temp0_minus_one = graph.cast_input_source::<Add>(temp0_store.value())?;
        if !ports_match(
            graph,
            values,
            (temp0_minus_one.lhs(), temp0_minus_one.rhs()),
            temp0_load.output_value(),
            -1,
        ) {
            return None;
        }

        // `temp0_not_zero := not temp0_eq_zero`
        let temp0_not_zero = graph.cast_input_source::<Not>(theta.condition().input())?;

        // `temp0_eq_zero := eq temp0_minus_one, int 0`
        let temp0_eq_zero = graph.cast_input_source::<Eq>(temp0_not_zero.input())?;
        if !ports_match(
            graph,
            values,
            (temp0_eq_zero.lhs(), temp0_eq_zero.rhs()),
            temp0_minus_one.value(),
            0,
        ) {
            return None;
        }

        Some(())
    }
}

fn ports_match(
    graph: &Rvsdg,
    values: &BTreeMap<OutputPort, Const>,
    (lhs, rhs): (InputPort, InputPort),
    value: OutputPort,
    literal: i32,
) -> bool {
    let (lhs_src, rhs_src) = (graph.input_source(lhs), graph.input_source(rhs));

    let (lhs_val, rhs_val) = (
        values.get(&lhs_src).and_then(Const::convert_to_i32),
        values.get(&rhs_src).and_then(Const::convert_to_i32),
    );

    (lhs_src == value && rhs_val == Some(literal)) || (lhs_val == Some(literal) && rhs_src == value)
}

/// Returns `true` if the given graph has no other nodes between its [`Start`] and [`End`] nodes
fn gamma_branch_is_empty(branch: &Rvsdg) -> bool {
    debug_assert_eq!(branch.start_nodes().len(), 1);
    debug_assert_eq!(branch.end_nodes().len(), 1);

    let start = branch.to_node::<Start>(branch.start_nodes()[0]);
    let end = branch.to_node::<End>(branch.end_nodes()[0]);

    branch.input_source(end.input_effect()) == start.effect()
}

impl Pass for SquareCell {
    fn pass_name(&self) -> &str {
        "square-cell"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.values.clear();
        self.changed = false;
    }

    fn report(&self) -> HashMap<&'static str, usize> {
        map! {
            "square loops" => self.squares_removed,
        }
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: i32) {
        let replaced = self.values.insert(int.value(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Int(value)));
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let (mut true_visitor, mut false_visitor) = (Self::new(), Self::new());
        let mut changed = false;

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[true_param, false_param]) in gamma.inputs().iter().zip(gamma.input_params())
        {
            let (_, source, _) = graph.get_input(input);

            if let Some(constant) = self.values.get(&source).copied() {
                let true_param = gamma.true_branch().to_node::<InputParam>(true_param);
                true_visitor
                    .values
                    .insert(true_param.output(), constant)
                    .debug_unwrap_none();

                let false_param = gamma.false_branch().to_node::<InputParam>(false_param);
                false_visitor
                    .values
                    .insert(false_param.output(), constant)
                    .debug_unwrap_none();
            }
        }

        changed |= true_visitor.visit_graph(gamma.true_mut());
        changed |= false_visitor.visit_graph(gamma.false_mut());

        if let Some(theta_id) = self.outer_gamma_is_candidate(&gamma) {
            let mut outmost_theta_visitor = Self::new();
            let mut outmost_theta = gamma.false_branch().to_node::<Theta>(theta_id).clone();

            // For each input into the theta region, if the input value is a known constant
            // then we should associate the input value with said constant
            for (input, param) in outmost_theta.invariant_input_pairs() {
                if let Some(constant) = false_visitor
                    .values
                    .get(&gamma.false_branch().input_source(input))
                    .cloned()
                {
                    outmost_theta_visitor
                        .values
                        .insert(param.output(), constant)
                        .debug_unwrap_none();
                }
            }
            changed |= outmost_theta_visitor.visit_graph(outmost_theta.body_mut());

            if let Some(candidate) = false_visitor
                .outmost_theta_is_candidate(&outmost_theta_visitor.values, &outmost_theta)
            {
                let OutmostTheta {
                    inner_gamma_id,
                    src_ptr,
                    temp0_ptr,
                    temp1_ptr,
                } = candidate;

                let inner_gamma = outmost_theta.body().to_node::<Gamma>(inner_gamma_id);
                let mut inner_gamma_visitor = Self::new();

                for (&input, &[_, param]) in
                    inner_gamma.inputs().iter().zip(inner_gamma.input_params())
                {
                    if let Some(constant) = outmost_theta_visitor
                        .values
                        .get(&outmost_theta.body().input_source(input))
                        .cloned()
                    {
                        let param = inner_gamma.false_branch().to_node::<InputParam>(param);

                        inner_gamma_visitor
                            .values
                            .insert(param.output(), constant)
                            .debug_unwrap_none();
                    }
                }
                inner_gamma_visitor.visit_graph(&mut inner_gamma.false_branch().clone());

                if let Some(InnerGamma { inner_theta_id }) =
                    inner_gamma_visitor.inner_gamma_is_candidate(inner_gamma)
                {
                    let inner_theta = inner_gamma.false_branch().to_node::<Theta>(inner_theta_id);
                    let mut inner_theta_visitor = Self::new();

                    for (input, param) in inner_theta.invariant_input_pairs() {
                        if let Some(constant) = inner_gamma_visitor
                            .values
                            .get(&inner_gamma.false_branch().input_source(input))
                            .cloned()
                        {
                            inner_theta_visitor
                                .values
                                .insert(param.output(), constant)
                                .debug_unwrap_none();
                        }
                    }
                    inner_theta_visitor.visit_graph(&mut inner_theta.body().clone());

                    if let Some(()) = inner_theta_visitor
                        .inner_theta_is_candidate(inner_theta, &inner_theta_visitor.values)
                    {
                        // FIXME: Check that src/temp0/temp1 pointers are the same

                        // FIXME: ???
                        let mut get_input = |output| {
                            outmost_theta_visitor
                                .values
                                .get(&output)
                                .and_then(Const::convert_to_i32)
                                .map(|int| graph.int(int).value())
                                .or_else(|| {
                                    outmost_theta
                                        .invariant_input_pairs()
                                        .find_map(|(port, input)| {
                                            if input.output() == output {
                                                Some(gamma.false_branch().input_source(port))
                                            } else {
                                                None
                                            }
                                        })
                                        .and_then(|output| {
                                            self.values
                                                .get(&output)
                                                .and_then(Const::convert_to_i32)
                                                .map(|int| graph.int(int).value())
                                                .or_else(|| {
                                                    gamma
                                                        .inputs()
                                                        .iter()
                                                        .zip(gamma.input_params())
                                                        .find_map(|(&port, &[_, input])| {
                                                            let input =
                                                                gamma
                                                                    .false_branch()
                                                                    .to_node::<InputParam>(input);

                                                            if input.output() == output {
                                                                Some(graph.input_source(port))
                                                            } else {
                                                                None
                                                            }
                                                        })
                                                })
                                        })
                                })
                        };

                        // ```
                        // // Load the source value
                        // src_val := load src_ptr
                        //
                        // // Multiply the source value by itself
                        // src_squared := mul src_val, src_val
                        //
                        // // Store the squared value into the source cell
                        // store src_ptr, src_squared
                        //
                        // // Zero out temp0 and temp1's cells
                        // store temp0_ptr, int 0
                        // store temp1_ptr, int 0
                        // ```
                        if let (Some(src_ptr), Some(temp0_ptr), Some(temp1_ptr)) = (
                            get_input(src_ptr),
                            get_input(temp0_ptr),
                            get_input(temp1_ptr),
                        ) {
                            tracing::trace!(
                                ?src_ptr,
                                ?temp0_ptr,
                                ?temp1_ptr,
                                "found square cell candidate in gamma node {:?}",
                                gamma.node(),
                            );

                            let input_effect = graph.input_source(gamma.effect_in());

                            // `src_val := load src_ptr`
                            let src_load = graph.load(temp0_ptr, input_effect);

                            // `src_squared := mul src_val, src_val`
                            let src_squared =
                                graph.mul(src_load.output_value(), src_load.output_value());

                            // `store src_ptr, src_squared`
                            let src_store =
                                graph.store(src_ptr, src_squared.value(), src_load.output_effect());

                            let zero = graph.int(0).value();

                            // `store temp0_ptr, int 0`
                            let zero_temp0 =
                                graph.store(temp0_ptr, zero, src_store.output_effect());

                            // `store temp1_ptr, int 0`
                            let zero_temp1 =
                                graph.store(temp1_ptr, zero, zero_temp0.output_effect());

                            // Wire the final store into the gamma's output effect
                            graph.rewire_dependents(gamma.effect_out(), zero_temp1.output_effect());

                            // FIXME: Patch up any output ports

                            graph.remove_node(gamma.node());
                            self.squares_removed += 1;
                            self.changed();

                            return;
                        }
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

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (input, param) in theta.invariant_input_pairs() {
            if let Some(constant) = self.values.get(&graph.input_source(input)).cloned() {
                visitor
                    .values
                    .insert(param.output(), constant)
                    .debug_unwrap_none();
            }
        }

        changed |= visitor.visit_graph(theta.body_mut());

        if changed {
            graph.replace_node(theta.node(), theta);
            self.changed();
        }
    }
}

impl Default for SquareCell {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct OutmostTheta {
    inner_gamma_id: NodeId,
    src_ptr: OutputPort,
    temp0_ptr: OutputPort,
    temp1_ptr: OutputPort,
}

impl OutmostTheta {
    const fn new(
        inner_gamma_id: NodeId,
        src_ptr: OutputPort,
        temp0_ptr: OutputPort,
        temp1_ptr: OutputPort,
    ) -> Self {
        Self {
            inner_gamma_id,
            src_ptr,
            temp0_ptr,
            temp1_ptr,
        }
    }
}

#[derive(Debug)]
struct InnerGamma {
    inner_theta_id: NodeId,
}

impl InnerGamma {
    const fn new(inner_theta_id: NodeId) -> Self {
        Self { inner_theta_id }
    }
}

test_opts! {
    square_input,
    passes = [SquareCell::new()],
    tape_size = 20,
    input = [10],
    output = [100],
    |graph, effect| {
        let source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/square.bf"));

        let ptr = graph.int(0);
        let (_ptr, effect) = compile_brainfuck_into(source, graph, ptr.value(), effect);
        effect
    },
}
