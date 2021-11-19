use crate::{
    graph::{
        End, Gamma, InputParam, Int, Load, Node, NodeExt, NodeId, OutputPort, Rvsdg, Start, Store,
        Theta,
    },
    ir::Const,
    passes::Pass,
    utils::{AssertNone, HashMap},
};

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
    values: HashMap<OutputPort, Const>,
    squares_removed: usize,
}

impl SquareCell {
    pub fn new() -> Self {
        Self {
            values: HashMap::with_hasher(Default::default()),
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
    fn outmost_theta_is_candidate(&self, theta: &Theta) -> Option<OutmostCandidate> {
        todo!()
    }
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
        "shift-cell"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.values.clear();
        self.changed = false;
    }

    fn report(&self) {
        tracing::info!(
            "{} removed {} cell shift motifs",
            self.pass_name(),
            self.squares_removed,
        );
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

            if let Some(constant) = self.values.get(&source).cloned() {
                let true_param = gamma.true_branch().to_node::<InputParam>(true_param);
                true_visitor
                    .values
                    .insert(true_param.output(), constant.clone())
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
            let outmost_theta = gamma.false_branch().to_node::<Theta>(theta_id);
            if let Some(candidate) = false_visitor.outmost_theta_is_candidate(outmost_theta) {
                let OutmostCandidate {
                    inner_gamma_id,
                    src_ptr,
                    temp0_ptr,
                    temp1_ptr,
                } = candidate;

                todo!()
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
        for (input, param) in theta.input_pairs() {
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
struct OutmostCandidate {
    inner_gamma_id: NodeId,
    src_ptr: OutputPort,
    temp0_ptr: OutputPort,
    temp1_ptr: OutputPort,
}

impl OutmostCandidate {
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
