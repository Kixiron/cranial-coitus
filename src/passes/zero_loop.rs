use crate::{
    graph::{
        AddOrSub, Bool, Byte, EdgeKind, End, Eq, Gamma, InputParam, InputPort, Int, Load, Neq,
        Node, NodeExt, OutputPort, Rvsdg, Start, Store, Theta,
    },
    passes::{
        utils::{ChangeReport, ConstantStore},
        Pass,
    },
    values::{Cell, Ptr},
};

pub struct ZeroLoop {
    changed: bool,
    values: ConstantStore,
    zero_gammas_removed: usize,
    zero_loops_removed: usize,
    tape_len: u16,
}

impl ZeroLoop {
    pub fn new(tape_len: u16) -> Self {
        Self {
            changed: false,
            values: ConstantStore::new(tape_len),
            zero_gammas_removed: 0,
            zero_loops_removed: 0,
            tape_len,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    /// See if the theta node matches our zero loop patterns
    ///
    /// Zero via overflow:
    ///
    /// ```
    /// do {
    ///   cell_value = load cell_ptr
    ///   plus_one = add cell_value, int 1
    ///   store cell_ptr, plus_one
    ///   not_zero = neq cell_value, int 1
    /// } while { not_zero }
    /// ```
    ///
    /// Zero via subtraction:
    ///
    /// ```
    /// do {
    ///   cell_value = load cell_ptr
    ///   minus_one = sub cell_value, int 1
    ///   store cell_ptr, minus_one
    ///   not_zero = neq cell_value, int 1
    /// } while { not_zero }
    /// ```
    ///
    /// In both of these cases we can replace the entire loop with a simple
    /// zero store.
    ///
    /// Don't be fooled by the `neq cell_value, 1`s, those are caused by
    /// another rewrite and still effectively means `eq cell_value, 0`
    /// for our use case since this is a do-while loop, meaning that on the
    /// iteration where it is true (and cell_value is one), one more
    /// subtraction/addition and store cycle would still happen.
    ///
    /// ```
    /// store cell_ptr, int 0
    /// ```
    fn is_zero_loop(
        &self,
        theta: &Theta,
        body_values: &ConstantStore,
    ) -> Option<Result<Ptr, InputPort>> {
        let graph = theta.body();

        // The theta's start node
        let start = theta.start_node();

        // cell_value = load cell_ptr
        let cell_value = graph.cast_output_dest::<Load>(start.effect())?;
        let cell_ptr = graph.input_source(cell_value.ptr());

        // plus_one = add cell_value, int 1
        // minus_one = sub cell_value, int 1
        let add_or_sub = AddOrSub::cast_output_dest(graph, cell_value.output_value())?;

        // store cell_ptr, plus_one
        // store cell_ptr, minus_one
        let store = graph.cast_output_dest::<Store>(cell_value.output_effect())?;
        if graph.input_source(store.ptr()) != cell_ptr
            || graph.input_source(store.value()) != add_or_sub.value()
        {
            return None;
        }

        // not_zero = neq cell_value, int 1
        // not_zero = neq plus_one, int 0
        // not_zero = neq minus_one, int 0
        let not_zero = graph.cast_input_source::<Neq>(theta.condition().input())?;
        {
            let lhs = graph.input_source(not_zero.lhs());
            let rhs_value = body_values.ptr(graph.input_source(not_zero.rhs()))?;

            // not_zero = neq cell_value, int 1
            let is_neq_one = lhs == cell_value.output_value() && rhs_value == 1;

            // not_zero = neq plus_one, int 0
            // not_zero = neq minus_one, int 0
            let is_neq_zero = lhs == add_or_sub.value() && rhs_value == 0;

            if !(is_neq_one || is_neq_zero) {
                return None;
            }
        }

        // The theta's end node
        let _end = graph.cast_output_dest::<End>(store.output_effect())?;

        // Now that we've successfully matched the motif, we can extract the data of cell_ptr
        let cell_ptr_value = match body_values.ptr(cell_ptr) {
            Some(ptr) => Ok(ptr),
            None => {
                // FIXME: We're pretty strict right now and only take inputs as "dynamic addresses",
                //        is it possible that this could be relaxed?
                let input = graph.cast_input_source::<InputParam>(cell_value.ptr())?;

                // Find the port that the input param refers to
                let input_port = theta
                    .input_pairs()
                    .find_map(|(port, param)| (param.node() == input.node()).then(|| port))?;

                Err(input_port)
            }
        };

        Some(cell_ptr_value)
    }

    /// If a gamma node is equivalent to one of our motifs,
    /// transform it into an unconditional zero store
    ///
    /// ```
    /// store ptr, int 0
    /// ```
    fn is_zero_gamma_store(
        &self,
        graph: &Rvsdg,
        false_values: &ConstantStore,
        gamma: &Gamma,
    ) -> Option<Result<Ptr, OutputPort>> {
        self.gamma_store_motif_1(graph, false_values, gamma)
            .or_else(|| self.gamma_store_motif_2(graph, false_values, gamma))
    }

    /// ```
    /// value = load ptr
    /// value_is_zero = eq value, int 0
    /// if value_is_zero {
    ///
    /// } else {
    ///   store ptr, int 0
    /// }
    /// ```
    #[allow(clippy::logic_bug)]
    fn gamma_store_motif_1(
        &self,
        graph: &Rvsdg,
        false_values: &ConstantStore,
        gamma: &Gamma,
    ) -> Option<Result<Ptr, OutputPort>> {
        // value_is_zero = eq value, int 0
        let eq_zero = graph.cast_input_source::<Eq>(gamma.condition())?;
        let [lhs, rhs] = [
            graph.input_source(eq_zero.lhs()),
            graph.input_source(eq_zero.rhs()),
        ];

        let (lhs_zero, rhs_zero) = (self.values.ptr_is_zero(lhs), self.values.ptr_is_zero(rhs));

        // If the eq doesn't fit the pattern of `value_is_zero = eq value, int 0` this isn't a candidate
        let value = if lhs_zero && rhs_zero || !lhs_zero && rhs_zero {
            graph.cast_parent::<_, Load>(lhs)?
        } else if lhs_zero && !rhs_zero {
            graph.cast_parent::<_, Load>(rhs)?
        } else {
            return None;
        };

        let source = graph.input_source(value.ptr());
        let target_ptr = self.values.ptr(source).ok_or(source);

        let start_effect = gamma
            .true_branch()
            .to_node::<Start>(gamma.starts()[0])
            .effect();
        let end_effect = gamma
            .true_branch()
            .to_node::<End>(gamma.ends()[0])
            .input_effect();

        // Make sure the true branch is empty
        let true_branch_is_empty =
            gamma.true_branch().output_dest(start_effect).next() == Some(end_effect);
        if !true_branch_is_empty {
            return None;
        }

        let start = gamma.false_branch().to_node::<Start>(gamma.starts()[1]);

        // store _ptr, int 0
        let store = gamma
            .false_branch()
            .cast_output_dest::<Store>(start.effect())?;

        // If the stored value isn't zero this isn't a candidate
        if !false_values.ptr_is_zero(gamma.false_branch().input_source(store.value())) {
            return None;
        }

        // Store should be the only/last thing in the branch
        let _end = gamma
            .false_branch()
            .cast_output_dest::<End>(store.output_effect())?;

        tracing::debug!("gamma store motif 1 matched");
        Some(target_ptr)
    }

    /// ```
    /// value_is_zero = eq value, int 0
    /// store ptr, value
    /// if value_is_zero {
    ///
    /// } else {
    ///   store ptr, int 0
    /// }
    /// ```
    #[allow(clippy::logic_bug)]
    fn gamma_store_motif_2(
        &self,
        graph: &Rvsdg,
        false_values: &ConstantStore,
        gamma: &Gamma,
    ) -> Option<Result<Ptr, OutputPort>> {
        // value_is_zero = eq value, int 0
        let eq_zero = graph.cast_input_source::<Eq>(gamma.condition())?;
        let [lhs, rhs] = [
            graph.input_source(eq_zero.lhs()),
            graph.input_source(eq_zero.rhs()),
        ];

        let (lhs_zero, rhs_zero) = (self.values.ptr_is_zero(lhs), self.values.ptr_is_zero(rhs));

        // If the eq doesn't fit the pattern of `value_is_zero = eq value, int 0` this isn't a candidate
        let value = if lhs_zero && rhs_zero || !lhs_zero && rhs_zero {
            lhs
        } else if lhs_zero && !rhs_zero {
            rhs
        } else {
            return None;
        };

        // store ptr, value
        let store = graph.cast_input_source::<Store>(gamma.input_effect())?;
        if graph.input_source(store.value()) != value {
            return None;
        }

        let source = graph.input_source(store.ptr());
        let target_ptr = self.values.ptr(source).ok_or(source);

        let start_effect = gamma
            .true_branch()
            .to_node::<Start>(gamma.starts()[0])
            .effect();
        let end_effect = gamma
            .true_branch()
            .to_node::<End>(gamma.ends()[0])
            .input_effect();

        // Make sure the true branch is empty
        let true_branch_is_empty =
            gamma.true_branch().output_dest(start_effect).next() == Some(end_effect);
        if !true_branch_is_empty {
            return None;
        }

        let start = gamma.false_branch().to_node::<Start>(gamma.starts()[1]);

        // store ptr, int 0
        let store = gamma
            .false_branch()
            .cast_output_dest::<Store>(start.effect())?;

        // If the stored value isn't zero this isn't a candidate
        if !false_values.ptr_is_zero(gamma.false_branch().input_source(store.value())) {
            return None;
        }

        // Store should be the only/last thing in the branch
        let _end = gamma
            .false_branch()
            .cast_output_dest::<End>(store.output_effect())?;

        tracing::debug!("gamma store motif 2 matched");
        Some(target_ptr)
    }
}

impl Pass for ZeroLoop {
    fn pass_name(&self) -> &str {
        "zero-loop"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.values.clear();
        self.changed = false;
    }

    fn report(&self) -> ChangeReport {
        map! {
            "zero loops" => self.zero_loops_removed,
            "zero gammas" => self.zero_gammas_removed,
        }
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        self.values.add(bool.value(), value);
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: Ptr) {
        self.values.add(int.value(), value);
    }

    fn visit_byte(&mut self, _graph: &mut Rvsdg, byte: Byte, value: Cell) {
        self.values.add(byte.value(), value);
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;

        let (mut true_visitor, mut false_visitor) =
            (Self::new(self.tape_len), Self::new(self.tape_len));
        self.values.gamma_inputs_into(
            &gamma,
            graph,
            &mut true_visitor.values,
            &mut false_visitor.values,
        );

        changed |= true_visitor.visit_graph(gamma.true_mut());
        self.zero_loops_removed += true_visitor.zero_loops_removed;
        self.zero_gammas_removed += true_visitor.zero_gammas_removed;

        changed |= false_visitor.visit_graph(gamma.false_mut());
        self.zero_loops_removed += false_visitor.zero_loops_removed;
        self.zero_gammas_removed += false_visitor.zero_gammas_removed;

        if let Some(target_ptr) = self.is_zero_gamma_store(graph, &false_visitor.values, &gamma) {
            let zero = graph.int(Ptr::zero(self.tape_len));
            let target_ptr = match target_ptr {
                Ok(ptr) if ptr.is_zero() => zero.value(),
                Ok(ptr) => graph.int(ptr).value(),
                Err(port) => port,
            };

            tracing::debug!(
                "detected that gamma {:?} is a zero store, replacing with a store of 0 to {:?}",
                gamma.node(),
                target_ptr,
            );

            let effect = graph.input_source(gamma.input_effect());
            let store = graph.store(target_ptr, zero.value(), effect);
            graph.rewire_dependents(gamma.output_effect(), store.output_effect());

            if gamma.outputs().len() == 1 {
                tracing::warn!("write comprehensive parameter rerouting for gamma nodes");
                graph.rewire_dependents(gamma.outputs()[0], target_ptr);
            } else {
                tracing::error!("write comprehensive parameter rerouting for gamma nodes");
            }

            graph.remove_node(gamma.node());
            self.zero_gammas_removed += 1;
            self.changed();
        } else if changed {
            graph.replace_node(gamma.node(), gamma);
            self.changed();
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut changed = false;

        let mut visitor = Self::new(self.tape_len);
        self.values
            .theta_invariant_inputs_into(&theta, graph, &mut visitor.values);

        changed |= visitor.visit_graph(theta.body_mut());
        self.zero_loops_removed += visitor.zero_loops_removed;
        self.zero_gammas_removed += visitor.zero_gammas_removed;

        if let Some(target_ptr) = self.is_zero_loop(&theta, &visitor.values) {
            tracing::debug!(
                "detected that theta {:?} is a zero loop, replacing with a store to {:?}",
                theta.node(),
                target_ptr,
            );

            // Get the theta's effect input
            let effect_source = graph.input_source(theta.input_effect().unwrap());

            let target_cell = match target_ptr {
                Ok(constant) => graph.int(constant).value(),
                Err(input_param) => graph.get_input(input_param).1,
            };

            // Create the zero store
            let zero = graph.int(Ptr::zero(self.tape_len));
            let store = graph.store(target_cell, zero.value(), effect_source);

            // Rewire the theta's ports
            graph.rewire_dependents(theta.output_effect().unwrap(), store.output_effect());

            for (input_port, param) in theta.input_pairs() {
                if let Some((Node::OutputParam(output), _, EdgeKind::Value)) =
                    theta.body().get_output(param.output())
                {
                    let output_port = theta.output_pairs().find_map(|(output_port, param)| {
                        (param.node() == output.node()).then(|| output_port)
                    });

                    if let Some(output_port) = output_port {
                        tracing::debug!(
                            "splicing theta input to output passthrough {:?}->{:?}",
                            input_port,
                            output_port,
                        );

                        graph.splice_ports(input_port, output_port);
                    }
                }
            }

            graph.remove_node(theta.node());
            self.zero_loops_removed += 1;
            self.changed();
        } else if changed {
            graph.replace_node(theta.node(), theta);
            self.changed();
        }
    }
}
