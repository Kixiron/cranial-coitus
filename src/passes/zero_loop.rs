use crate::{
    graph::{
        Add, Bool, EdgeKind, End, Eq, Gamma, InputPort, Int, Load, Node, NodeExt, Not, OutputPort,
        Rvsdg, Start, Store, Sub, Theta,
    },
    passes::{utils::ConstantStore, Pass},
    utils::HashMap,
    values::Ptr,
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
    ///   plus_one = add cell, int 1
    ///   store _cell_ptr, plus_one
    ///   is_zero = eq plus_one, int 0
    ///   not_zero = not is_zero
    /// } while { _not_zero }
    /// ```
    ///
    /// Zero via subtraction:
    ///
    /// ```
    /// do {
    ///   cell_value = load cell_ptr
    ///   minus_one = sub cell_value, int 1
    ///   store cell_ptr, minus_one
    ///   is_zero = eq minus_one, int 0
    ///   not_zero = not is_zero
    /// } while { not_zero }
    /// ```
    ///
    /// In both of these cases we can replace the entire loop with a simple
    /// zero store
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

        // The node after the effect must be a load
        let load = graph.cast_output_dest::<Load>(start.effect())?;

        let (target_node, source, _) = graph.get_input(load.ptr());

        // We can zero out constant addresses or a dynamically generated ones
        let target_ptr = match body_values.ptr(source) {
            Some(ptr) => Ok(ptr),
            None => {
                // FIXME: We're pretty strict right now and only take inputs as "dynamic addresses",
                //        is it possible that this could be relaxed?
                let input = target_node.as_input_param()?;

                // Find the port that the input param refers to
                let input_port = theta
                    .input_pairs()
                    .find_map(|(port, param)| (param.node() == input.node()).then(|| port))?;

                Err(input_port)
            }
        };

        // TODO: Refactor
        let value = if let Some(add) = graph.cast_output_dest::<Add>(load.output_value()) {
            let (lhs, rhs) = (graph.input_source(add.lhs()), graph.input_source(add.rhs()));

            // Make sure that one of the add's operands is the loaded cell and the other is 1 or -1
            if lhs == load.output_value() {
                let value = body_values.ptr(rhs)?;

                // Any odd integer will eventually converge to zero
                if value.is_even() {
                    // TODO: If value is divisible by 2 this loop is finite if the loaded value is even.
                    //       Otherwise if the loaded value is odd or `value` is zero, this loop is infinite.
                    //       Lastly, if both the loaded value and `value` are zero, this is an entirely redundant
                    //       loop and we can remove it entirely for nothing, not even a store
                    return None;
                }
            } else if rhs == load.output_value() {
                let value = body_values.ptr(lhs)?;

                // Any odd integer will eventually converge to zero
                if value.is_even() {
                    // TODO: If value is divisible by 2 this loop is finite if the loaded value is even.
                    //       Otherwise if the loaded value is odd or `value` is zero, this loop is infinite.
                    //       Lastly, if both the loaded value and `value` are zero, this is an entirely redundant
                    //       loop and we can remove it entirely for nothing, not even a store
                    return None;
                }
            } else {
                return None;
            }

            add.value()
        } else if let Some(sub) = graph.cast_output_dest::<Sub>(load.output_value()) {
            let (lhs, rhs) = (graph.input_source(sub.lhs()), graph.input_source(sub.rhs()));

            // Make sure that one of the add's operands is the loaded cell and the other is 1 or -1
            if lhs == load.output_value() {
                let value = body_values.ptr(rhs)?;

                // Any odd integer will eventually converge to zero
                if value.is_even() {
                    // TODO: If value is divisible by 2 this loop is finite if the loaded value is even.
                    //       Otherwise if the loaded value is odd or `value` is zero, this loop is infinite.
                    //       Lastly, if both the loaded value and `value` are zero, this is an entirely redundant
                    //       loop and we can remove it entirely for nothing, not even a store
                    return None;
                }
            } else if rhs == load.output_value() {
                let value = body_values.ptr(lhs)?;

                // Any odd integer will eventually converge to zero
                if value.is_even() {
                    // TODO: If value is divisible by 2 this loop is finite if the loaded value is even.
                    //       Otherwise if the loaded value is odd or `value` is zero, this loop is infinite.
                    //       Lastly, if both the loaded value and `value` are zero, this is an entirely redundant
                    //       loop and we can remove it entirely for nothing, not even a store
                    return None;
                }
            } else {
                return None;
            }

            sub.value()
        } else {
            return None;
        };

        let store = graph.cast_output_dest::<Store>(load.output_effect())?;
        if graph.input_source(store.value()) != value {
            return None;
        }

        let eq = graph
            .get_outputs(value)
            .find_map(|(node, ..)| node.as_eq())?;

        let (lhs, rhs) = (graph.input_source(eq.lhs()), graph.input_source(eq.rhs()));

        // Make sure that one of the eq's operands is the added val and the other is 0
        if lhs == value {
            let value = body_values.ptr(rhs)?;

            if value != 0 {
                return None;
            }
        } else if rhs == value {
            let value = body_values.ptr(lhs)?;

            if value != 0 {
                return None;
            }
        } else {
            return None;
        }

        let not = graph.cast_output_dest::<Not>(eq.value())?;

        // Make sure the `(value ± 1) != 0` expression is the theta's condition
        if graph.output_dest_id(not.value())? != theta.condition().node() {
            return None;
        }

        let _end = graph.cast_output_dest::<End>(store.output_effect())?;

        Some(target_ptr)
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
        let eq_zero = graph.get_input(gamma.condition()).0.as_eq()?;
        let [lhs, rhs] = [
            graph.get_input(eq_zero.lhs()),
            graph.get_input(eq_zero.rhs()),
        ];

        let (lhs_zero, rhs_zero) = (
            self.values.ptr_is_zero(lhs.1),
            self.values.ptr_is_zero(rhs.1),
        );

        // If the eq doesn't fit the pattern of `value_is_zero = eq value, int 0` this isn't a candidate
        let value = if lhs_zero && rhs_zero || !lhs_zero && rhs_zero {
            lhs.0
        } else if lhs_zero && !rhs_zero {
            rhs.0
        } else {
            return None;
        }
        .as_load()?;

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
        if false_values.ptr_is_zero(gamma.false_branch().input_source(store.value())) {
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
            graph.get_input(eq_zero.lhs()),
            graph.get_input(eq_zero.rhs()),
        ];

        let (lhs_zero, rhs_zero) = (
            self.values.ptr_is_zero(lhs.1),
            self.values.ptr_is_zero(rhs.1),
        );

        // If the eq doesn't fit the pattern of `value_is_zero = eq value, int 0` this isn't a candidate
        let value = if lhs_zero && rhs_zero || !lhs_zero && rhs_zero {
            lhs.1
        } else if lhs_zero && !rhs_zero {
            rhs.1
        } else {
            return None;
        };

        // store _ptr, _value
        let store = graph.cast_input_source::<Store>(gamma.effect_in())?;
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

        // store _ptr, int 0
        let store = gamma
            .false_branch()
            .cast_output_dest::<Store>(start.effect())?;

        // If the stored value isn't zero this isn't a candidate
        if false_values.ptr_is_zero(gamma.false_branch().input_source(store.value())) {
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

    fn report(&self) -> HashMap<&'static str, usize> {
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
        changed |= false_visitor.visit_graph(gamma.false_mut());

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

            let effect = graph.input_source(gamma.effect_in());
            let store = graph.store(target_ptr, zero.value(), effect);
            graph.rewire_dependents(gamma.effect_out(), store.output_effect());

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
