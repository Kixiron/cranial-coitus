use tinyvec::{tiny_vec, TinyVec};

use crate::{
    graph::{Gamma, InputParam, InputPort, Int, Node, NodeExt, OutputPort, Rvsdg, Theta},
    ir::Const,
    passes::Pass,
    utils::{AssertNone, HashMap},
};

pub struct DuplicateCell {
    changed: bool,
    values: HashMap<OutputPort, Const>,
}

impl DuplicateCell {
    pub fn new() -> Self {
        Self {
            values: HashMap::with_hasher(Default::default()),
            changed: false,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
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
    /// // TODO: Technically this should add src_val to the values
    /// //       of the respective cells
    /// store d1_ptr, src_val
    /// store d2_ptr, src_val
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
        let (d1_ptr, _, d1_effect) = self.load_add_store(graph, values, start.effect(), 1)?;

        // Get the load-add-store sequence of the second duplicate cell
        // ```
        // d2_val := load d2_ptr
        // d2_inc := add d2_val, int 1
        // store d1_ptr, d2_inc
        // ```
        let (d2_ptr, _, d2_effect) = self.load_add_store(graph, values, d1_effect, 1)?;

        // Get the load-add-store sequence of the source cell
        // ```
        // src_val := load src_ptr
        // src_dec := add src_val, int -1
        // store src_ptr, src_src_dec
        // ```
        let (src_ptr, src_dec, src_effect) = self.load_add_store(graph, values, d2_effect, -1)?;

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

        // Yay, all of our validation is done and we've determined that this is
        // indeed a shift motif
        Some(DuplicateCandidate::new(src_ptr, tiny_vec![d1_ptr, d2_ptr]))
    }

    /// Matches a load-add-store motif, returns the `OutputPort` of the
    /// target pointer, the `OutputPort` of the add's value and the
    /// `OutputPort` of the last effect
    fn load_add_store(
        &self,
        graph: &Rvsdg,
        values: &HashMap<OutputPort, Const>,
        last_effect: OutputPort,
        add_value: i32,
    ) -> Option<(OutputPort, OutputPort, OutputPort)> {
        // Get the load
        // `val := load ptr`
        let load = graph.get_output(last_effect)?.0.as_load()?;
        let ptr = graph.input_source(load.ptr());

        // Get the store
        // `store ptr, val_add`
        let store = graph.get_output(load.effect())?.0.as_store()?;

        // Make sure the addresses are the same
        if graph.input_source(store.ptr()) != ptr {
            return None;
        }

        // Get the add
        // `val_add := add val, int 1`
        let val_add = graph.get_input(store.value()).0.as_add()?;

        // Make sure that the add is in the proper form, one side should be
        // `add_value` and the other should be the loaded value
        self.compare_operands(
            graph,
            values,
            val_add.lhs(),
            val_add.rhs(),
            load.value(),
            add_value,
        )?;

        Some((ptr, val_add.value(), store.effect()))
    }

    fn compare_operands(
        &self,
        graph: &Rvsdg,
        values: &HashMap<OutputPort, Const>,
        lhs: InputPort,
        rhs: InputPort,
        operand: OutputPort,
        constant: i32,
    ) -> Option<()> {
        let (lhs_operand, rhs_operand) = (graph.input_source(lhs), graph.input_source(rhs));

        if lhs_operand == operand {
            if values.get(&rhs_operand)?.as_int()? != constant {
                return None;
            }
        } else if rhs_operand == operand {
            if values.get(&lhs_operand)?.as_int()? != constant {
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

        if let Some(DuplicateCandidate { src_ptr, dest_ptrs }) =
            self.theta_is_candidate(&visitor.values, &theta)
        {
            tracing::debug!(
                "found theta that composes to a duplicate loop copying from {:?} to {:?}",
                src_ptr,
                dest_ptrs,
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

            let dest_ptr_inputs: Option<TinyVec<[_; 2]>> =
                dest_ptrs.iter().fold(Some(TinyVec::new()), |ptrs, &ptr| {
                    ptrs.and_then(|mut ptrs| {
                        ptrs.push(get_theta_input(ptr)?);
                        Some(ptrs)
                    })
                });

            if let (Some(src_ptr), Some(dest_ptrs)) = (get_theta_input(src_ptr), dest_ptr_inputs) {
                let input_effect = graph.input_source(theta.input_effect().unwrap());

                // Load the source value
                let src_val = graph.load(src_ptr, input_effect);

                // Store the source value into the destination cells
                let mut last_effect = src_val.effect();
                for dest_ptr in dest_ptrs {
                    let store_src_to_dest = graph.store(dest_ptr, src_val.value(), last_effect);
                    last_effect = store_src_to_dest.effect();
                }

                // Unconditionally store 0 to the destination cell
                let zero = graph.int(0);
                let zero_dest_cell = graph.store(src_ptr, zero.value(), last_effect);

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
                    "failed to optimize duplicate loop copying from {:?} to {:?}, \
                    the pointers were not in the expected form yet",
                    src_ptr,
                    dest_ptrs,
                );
            }
        }

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
