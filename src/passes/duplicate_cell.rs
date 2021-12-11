use crate::{
    graph::{
        Add, Gamma, InputParam, InputPort, Int, Node, NodeExt, OutputPort, Rvsdg, Start, Sub, Theta,
    },
    ir::Const,
    passes::{utils::BinOp, Pass},
    utils::{AssertNone, HashMap},
};
use tinyvec::{tiny_vec, TinyVec};

pub struct DuplicateCell {
    changed: bool,
    // TODO: Use ConstantStore
    values: HashMap<OutputPort, Const>,
    duplicates_removed: usize,
}

impl DuplicateCell {
    pub fn new() -> Self {
        Self {
            values: HashMap::with_hasher(Default::default()),
            changed: false,
            duplicates_removed: 0,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    fn visit_gamma_theta(
        &mut self,
        graph: &mut Rvsdg,
        theta: &mut Theta,
    ) -> (bool, Option<(DuplicateCandidate, Self)>) {
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

        (
            changed,
            self.theta_is_candidate(&visitor.values, theta)
                .map(|candidate| (candidate, visitor)),
        )
    }

    fn inline_duplicate_gamma(
        &mut self,
        gamma: &Gamma,
        theta: &Theta,
        visitor: Self,
        graph: &mut Rvsdg,
        theta_graph: &Rvsdg,
        duplicate_candidate: DuplicateCandidate,
    ) -> bool {
        let DuplicateCandidate { src_ptr, dest_ptrs } = duplicate_candidate;

        tracing::debug!(
            "found theta that composes to a duplicate loop copying from {:?} to {:?}",
            src_ptr,
            dest_ptrs,
        );

        // FIXME: ???
        let mut get_gamma_input = |output| {
            visitor
                .values
                .get(&output)
                .and_then(Const::convert_to_u32)
                .map(|int| graph.int(int).value())
                .or_else(|| {
                    theta
                        .invariant_input_pairs()
                        .find_map(|(port, input)| {
                            if input.output() == output {
                                Some(theta_graph.get_input(port).1)
                            } else {
                                None
                            }
                        })
                        .and_then(|output| {
                            self.values
                                .get(&output)
                                .and_then(Const::convert_to_u32)
                                .map(|int| graph.int(int).value())
                                .or_else(|| {
                                    gamma.inputs().iter().zip(gamma.input_params()).find_map(
                                        |(&port, &[_, input])| {
                                            let input =
                                                gamma.false_branch().to_node::<InputParam>(input);

                                            if input.output() == output {
                                                Some(graph.get_input(port).1)
                                            } else {
                                                None
                                            }
                                        },
                                    )
                                })
                        })
                })
        };

        let dest_ptr_inputs: Option<TinyVec<[_; 2]>> = dest_ptrs
            .iter()
            .fold(Some(TinyVec::new()), |ptrs, &ptr| {
                ptrs.and_then(|mut ptrs| {
                    ptrs.push(get_gamma_input(ptr)?);
                    Some(ptrs)
                })
            })
            .filter(|inputs| inputs.len() == dest_ptrs.len());

        if let (Some(src_ptr), Some(dest_ptrs)) = (get_gamma_input(src_ptr), dest_ptr_inputs) {
            let input_effect = graph.input_source(gamma.effect_in());

            // Load the source value
            let src_val = graph.load(src_ptr, input_effect);

            // Store the source value into the destination cells
            let mut last_effect = src_val.output_effect();

            // ```
            // d1_val = load d1_ptr
            // d1_sum = add d1_val, src_val
            // store d1_ptr, d1_sum
            // ```
            for dest_ptr in dest_ptrs {
                let load = graph.load(dest_ptr, last_effect);
                let sum = graph.add(load.output_value(), src_val.output_value());
                let store = graph.store(dest_ptr, sum.value(), load.output_effect());

                last_effect = store.output_effect();
            }

            // Unconditionally store 0 to the source cell
            let zero = graph.int(0);
            let zero_src_cell = graph.store(src_ptr, zero.value(), last_effect);

            // Wire the final store into the gamma's output effect
            graph.rewire_dependents(gamma.effect_out(), zero_src_cell.output_effect());

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

            graph.remove_node(gamma.node());
            self.changed();

            true
        } else {
            tracing::trace!(
                "failed to optimize duplicate loop copying from {:?} to {:?}, \
                the pointers were not in the expected form yet",
                src_ptr,
                dest_ptrs,
            );

            false
        }
    }

    /// ```
    /// // TODO: Use psi nodes to account for all orientations of this
    /// // TODO: Allow variadic numbers of duplicate targets
    /// do {
    ///     // Increment the first duplicate target
    ///     d1_val := load d1_ptr
    ///     d1_inc := add d1_val, int 1
    ///     store d1_ptr, d1_inc
    ///
    ///     // Increment the second duplicate target
    ///     d2_val := load d2_ptr
    ///     d2_inc := add d2_val, int 1
    ///     store d2_ptr, d2_inc
    ///
    ///     // Decrement the source cell
    ///     src_val := load src_ptr
    ///     src_dec := add src_val, int -1
    ///     store src_ptr, src_dec
    ///
    ///     // Test if the source cell is zero
    ///     src_is_zero := eq src_dec, int 0
    ///     src_is_non_zero := not v87
    /// } while { src_is_non_zero }
    /// ```
    ///
    /// After conversion, this should be the generated code:
    /// ```
    /// // Load the source value
    /// src_val := load src_ptr
    ///
    /// // Store the source value into the duplicate cells
    /// d1_val = load d1_ptr
    /// d1_sum = add d1_val, src_val
    /// store d1_ptr, d1_sum
    ///
    /// d2_val = load d2_ptr
    /// d2_sum = add d2_val, src_val
    /// store d2_ptr, d2_sum
    ///
    /// // Zero out the source cell
    /// store src_ptr, int 0
    /// ```
    fn theta_is_candidate(
        &self,
        values: &HashMap<OutputPort, Const>,
        theta: &Theta,
    ) -> Option<DuplicateCandidate> {
        let graph = theta.body();

        // Get the body's start node
        let start = theta.start_node();

        // Get the load-add-store sequence of the first duplicate cell
        // ```
        // d1_val := load d1_ptr
        // d1_inc := add d1_val, int 1
        // store d1_ptr, d1_inc
        // ```
        let (d1_ptr, _, d1_effect) = self.load_op_store::<Add>(graph, values, start.effect(), 1)?;

        // Get the load-add-store sequence of the second duplicate cell
        // ```
        // d2_val := load d2_ptr
        // d2_inc := add d2_val, int 1
        // store d1_ptr, d2_inc
        // ```
        let (d2_ptr, _, d2_effect) = self.load_op_store::<Add>(graph, values, d1_effect, 1)?;

        // Get the load-sub-store sequence of the source cell
        // ```
        // src_val := load src_ptr
        // src_dec := sub src_val, int 1
        // store src_ptr, src_src_dec
        // ```
        let (src_ptr, src_dec, src_effect) =
            self.load_op_store::<Sub>(graph, values, d2_effect, 1)?;

        // Get the end node of the theta body
        let end = theta.end_node();
        // Make sure the next effect is the end node
        if src_effect != graph.input_source(end.input_effect()) {
            return None;
        }

        // Get the theta's condition
        // ```
        // src_is_zero := eq src_dec, int 0
        // src_is_non_zero := not v87
        // ```
        let condition = theta.condition();

        // `src_is_non_zero := not v87`
        let src_is_non_zero = graph.input_source_node(condition.input()).as_not()?;

        // `src_is_zero := eq src_dec, int 0`
        let src_is_zero = graph.input_source_node(src_is_non_zero.input()).as_eq()?;
        self.compare_operands(
            graph,
            values,
            src_is_zero.lhs(),
            src_is_zero.rhs(),
            src_dec,
            0,
        )?;

        if src_ptr == d1_ptr || src_ptr == d2_ptr || d1_ptr == d2_ptr {
            return None;
        }

        // Yay, all of our validation is done and we've determined that this is
        // indeed a shift motif
        Some(DuplicateCandidate::new(src_ptr, tiny_vec![d1_ptr, d2_ptr]))
    }

    /// Matches a load-add-store motif, returns the `OutputPort` of the
    /// target pointer, the `OutputPort` of the add's value and the
    /// `OutputPort` of the last effect
    fn load_op_store<T>(
        &self,
        graph: &Rvsdg,
        values: &HashMap<OutputPort, Const>,
        last_effect: OutputPort,
        add_value: u32,
    ) -> Option<(OutputPort, OutputPort, OutputPort)>
    where
        T: BinOp,
        for<'a> &'a Node: TryInto<&'a T>,
    {
        // Get the load
        // `val := load ptr`
        let load = graph.get_output(last_effect)?.0.as_load()?;
        let ptr = graph.input_source(load.ptr());

        // Get the store
        // `store ptr, val_add`
        let store = graph.get_output(load.output_effect())?.0.as_store()?;

        // Make sure the addresses are the same
        if graph.input_source(store.ptr()) != ptr {
            return None;
        }

        // Get the add
        // `val_add := add val, int 1`
        let val_add = graph.cast_input_source::<T>(store.value())?;

        // Make sure that the add is in the proper form, one side should be
        // `add_value` and the other should be the loaded value
        self.compare_operands(
            graph,
            values,
            val_add.lhs(),
            val_add.rhs(),
            load.output_value(),
            add_value,
        )?;

        Some((ptr, val_add.value(), store.output_effect()))
    }

    fn compare_operands(
        &self,
        graph: &Rvsdg,
        values: &HashMap<OutputPort, Const>,
        lhs: InputPort,
        rhs: InputPort,
        operand: OutputPort,
        constant: u32,
    ) -> Option<()> {
        let (lhs_operand, rhs_operand) = (graph.input_source(lhs), graph.input_source(rhs));

        if lhs_operand == operand {
            if values.get(&rhs_operand)?.convert_to_u32()? != constant {
                return None;
            }
        } else if rhs_operand == operand {
            if values.get(&lhs_operand)?.convert_to_u32()? != constant {
                return None;
            }
        } else {
            return None;
        }

        Some(())
    }
}

impl Pass for DuplicateCell {
    fn pass_name(&self) -> &str {
        "duplicate-cell"
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
            "cell duplications" => self.duplicates_removed,
        }
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: u32) {
        let replaced = self.values.insert(int.value(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Int(value)));
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;
        let (mut true_visitor, mut false_visitor) = (Self::new(), Self::new());

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
            // TODO: Make sure the gamma's condition is an `eq (load ptr), int 0`
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

                let (gamma_changed, duplicate) =
                    false_visitor.visit_gamma_theta(gamma.false_mut(), &mut theta);

                tracing::trace!(
                    gamma = ?gamma.node(),
                    theta = ?theta.node(),
                    "gamma duplicate cell candidate: {:?}",
                    duplicate.as_ref().map(|(candidate, _)| candidate),
                );
                if let Some((candidate, theta_body_visitor)) = duplicate {
                    if self.inline_duplicate_gamma(
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
                            "inlined duplicate cell candidate",
                        );
                        self.changed();

                        self.duplicates_removed += 1;
                        return;
                    }
                }

                if gamma_changed {
                    gamma.false_mut().replace_node(theta.node(), theta);
                    changed = true;
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

impl Default for DuplicateCell {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct DuplicateCandidate {
    src_ptr: OutputPort,
    dest_ptrs: TinyVec<[OutputPort; 2]>,
}

impl DuplicateCandidate {
    fn new(src_ptr: OutputPort, dest_ptrs: TinyVec<[OutputPort; 2]>) -> Self {
        Self { src_ptr, dest_ptrs }
    }
}
