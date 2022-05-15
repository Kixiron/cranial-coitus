use crate::{
    graph::{Bool, Byte, Gamma, InputParam, Int, Load, NodeExt, Rvsdg, Store, Theta},
    passes::{
        utils::{ChangeReport, Changes, ConstantStore, MemoryCell, MemoryTape},
        Pass,
    },
    values::{Cell, Ptr},
};

/// Evaluates constant loads within the program
pub struct Mem2Reg {
    constants: ConstantStore,
    tape: MemoryTape,
    changes: Changes<4>,
}

// TODO: Propagate port places into subgraphs by adding input ports
impl Mem2Reg {
    pub fn new(tape_len: u16) -> Self {
        Self::new_inner(MemoryTape::zeroed(tape_len))
    }

    fn with_tape(tape: MemoryTape) -> Self {
        Self::new_inner(tape)
    }

    fn new_inner(tape: MemoryTape) -> Self {
        Self {
            constants: ConstantStore::new(tape.tape_len()),
            tape,
            changes: Changes::new([
                "const-loads",
                "loads-elided",
                "identical-stores",
                "dependent-loads",
            ]),
        }
    }
}

// TODO: Better analyze stores on the outs from thetas & gammas for fine-grained
//       tape invalidation
impl Pass for Mem2Reg {
    fn pass_name(&self) -> &'static str {
        "mem2reg"
    }

    fn did_change(&self) -> bool {
        self.changes.did_change()
    }

    fn report(&self) -> ChangeReport {
        self.changes.as_report()
    }

    fn reset(&mut self) {
        self.tape.zero();
        self.changes.reset();
        self.constants.clear();
    }

    fn visit_load(&mut self, graph: &mut Rvsdg, load: Load) {
        if let Some(offset) = self.constants.ptr(graph.input_source(load.ptr())) {
            match self.tape[offset] {
                MemoryCell::Cell(value) => {
                    tracing::debug!(
                        "replacing load from {} with value {:#04X}: {:?}",
                        offset,
                        value,
                        load,
                    );

                    let byte = graph.byte(value);
                    self.constants.add(byte.value(), value);

                    graph.splice_ports(load.input_effect(), load.output_effect());
                    graph.rewire_dependents(load.output_value(), byte.value());
                    graph.remove_node(load.node());

                    self.changes.inc::<"const-loads">();
                }

                MemoryCell::Port(output_port) => {
                    tracing::debug!(
                        "replacing load from {} with port {:?}: {:?}",
                        offset,
                        output_port,
                        load,
                    );

                    graph.splice_ports(load.input_effect(), load.output_effect());
                    graph.rewire_dependents(load.output_value(), output_port);
                    graph.remove_node(load.node());

                    self.changes.inc::<"loads-elided">();
                }

                MemoryCell::Unknown => {}
            }
        }
    }

    fn visit_store(&mut self, graph: &mut Rvsdg, store: Store) {
        if let Some(offset) = self.constants.ptr(graph.input_source(store.ptr())) {
            let value_source = graph.input_source(store.value());
            let stored_value = self.constants.cell(value_source);

            match (self.tape[offset].as_cell(), stored_value) {
                (Some(old), Some(new)) if old == new => {
                    self.tape[offset] = MemoryCell::Cell(new);

                    tracing::debug!("removing identical store {:?}", store);

                    graph.splice_ports(store.effect_in(), store.output_effect());
                    graph.remove_node(store.node());

                    self.changes.inc::<"identical-stores">();
                }

                (_, Some(value)) => self.tape[offset] = MemoryCell::Cell(value),

                (_, None) => {
                    self.tape[offset] = self
                        .constants
                        .cell(value_source)
                        .map_or(MemoryCell::Port(value_source), MemoryCell::Cell);
                }
            }
        } else {
            tracing::debug!("unknown store {:?}, invalidating tape", store);

            // Invalidate the whole tape
            self.tape.mystify();
        }

        // TODO: Move this to separate pass
        if let Some(&load) = graph.cast_output_dest::<Load>(store.output_effect()) {
            // If there's a sequence of
            //
            // ```
            // store _0, _1
            // _2 = load _0
            // ```
            //
            // we want to remove the redundant load since nothing has occurred
            // change the pointed-to value, changing this code to
            //
            // ```
            // store _0, _1
            // _2 = _1
            // ```
            //
            // but we don't have any sort of passthrough node currently (may
            // want to fix that, could be useful) we instead rewire all dependents on
            // the redundant load (`_2`) to instead point to the known value of the cell
            // (`_1`) transforming this code
            //
            // ```
            // store _0, _1
            // _2 = load _0
            // _3 = add _2, int 10
            // ```
            //
            // into this code
            //
            // ```
            // store _0, _1
            // _3 = add _1, int 10
            // ```
            if graph.input_source_id(load.ptr()) == graph.input_source_id(store.ptr()) {
                tracing::debug!(
                    "replaced dependent load with value {:?} (store: {:?})",
                    load,
                    store,
                );

                graph.splice_ports(load.input_effect(), load.output_effect());
                graph.rewire_dependents(load.output_value(), graph.input_source(store.value()));
                graph.remove_node(load.node());

                self.changes.inc::<"dependent-loads">();
            }
        }
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        self.constants.add(bool.value(), value);
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: Ptr) {
        self.constants.add(int.value(), value);
    }

    fn visit_byte(&mut self, _graph: &mut Rvsdg, byte: Byte, value: Cell) {
        self.constants.add(byte.value(), value);
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        // Don't propagate port places into subgraphs
        // FIXME: This is only a implementation limit right now,
        //        gamma nodes aren't iterative so they can have full
        //        access to the prior scope via input ports. We should
        //        add inputs as needed to bring branch-invariant code
        //        into said branches
        let tape = self.tape.mapped(|value| match value {
            MemoryCell::Cell(constant) => MemoryCell::Cell(constant),
            // TODO: Propagate input parameters
            _ => MemoryCell::Unknown,
        });

        // Both branches of the gamma node get the previous context, the changes they
        // create within it just are trickier to propagate
        let (mut true_visitor, mut false_visitor) =
            (Self::with_tape(tape.clone()), Self::with_tape(tape));

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        let inputs: Vec<_> = gamma
            .inputs()
            .iter()
            .zip(gamma.input_params())
            .map(|(&input, &params)| (input, params))
            .collect();

        true_visitor.constants.reserve(inputs.len());
        false_visitor.constants.reserve(inputs.len());

        for (input, [true_param, false_param]) in inputs {
            if let Some(value) = self.constants.get(graph.input_source(input)) {
                let param = gamma.true_branch().to_node::<InputParam>(true_param);
                true_visitor.constants.add(param.output(), value);

                let param = gamma.false_branch().to_node::<InputParam>(false_param);
                false_visitor.constants.add(param.output(), value);
            }
        }

        let mut changed = true_visitor.visit_graph(gamma.true_mut());
        self.changes.combine(&true_visitor.changes);

        changed |= false_visitor.visit_graph(gamma.false_mut());
        self.changes.combine(&false_visitor.changes);

        // Figure out if there's any stores within either of the gamma branches
        let contains_stores = gamma
            .false_branch()
            .try_for_each_transitive_node(|_, node| node.is_store())
            || gamma
                .true_branch()
                .try_for_each_transitive_node(|_, node| node.is_store());

        // Invalidate the whole tape if any stores occur within it
        // FIXME: We pessimistically clear out the *entire* tape, but
        //        ideally we'd only clear out the cells that are specifically
        //        stored to and only fall back to the pessimistic case in the
        //        event that we find a store to an unknown pointer
        if contains_stores {
            tracing::debug!("gamma node does contains stores, invalidating program tape");
            self.tape.mystify();
        } else {
            tracing::debug!("gamma node does no stores, not invalidating program tape");
        }

        if changed {
            graph.replace_node(gamma.node(), gamma);
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let body_contains_stores = theta
            .body()
            .try_for_each_transitive_node(|_, node| node.is_store());

        // If no stores occur within the theta's body then we can
        // propagate the current state of the tape into it and if not,
        // we can clear it out
        // FIXME: We pessimistically clear out the *entire* tape, but
        //        ideally we'd only clear out the cells that are specifically
        //        stored to and only fall back to the pessimistic case in the
        //        event that we find a store to an unknown pointer
        let tape = if body_contains_stores {
            MemoryTape::unknown(self.constants.tape_len())
        } else {
            self.tape.mapped(|place| match place {
                MemoryCell::Cell(cell) => MemoryCell::Cell(cell),
                // TODO: Propagate invariant constants
                _ => MemoryCell::Unknown,
            })
        };

        let mut visitor = Self::with_tape(tape);
        visitor.constants.reserve(theta.invariant_inputs_len());

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        // Note: We only propagate **invariant** inputs into the loop, propagating
        //       variant inputs requires dataflow information
        for (input, param) in theta.invariant_input_pairs() {
            if let Some(value) = self.constants.get(graph.input_source(input)) {
                visitor.constants.add(param.output(), value);
            }
        }

        let changed = visitor.visit_graph(theta.body_mut());
        self.changes.combine(&visitor.changes);

        // If any stores occur within the theta's body, invalidate the whole tape
        // FIXME: We pessimistically clear out the *entire* tape, but
        //        ideally we'd only clear out the cells that are specifically
        //        stored to and only fall back to the pessimistic case in the
        //        event that we find a store to an unknown pointer
        if body_contains_stores {
            tracing::debug!("theta body contains stores, invalidating program tape");
            self.tape.mystify();
        } else {
            tracing::debug!("theta body does no stores, not invalidating program tape");
        }

        if changed {
            graph.replace_node(theta.node(), theta);
        }
    }
}
