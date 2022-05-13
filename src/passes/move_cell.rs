use crate::{
    graph::{
        AddOrSub, Byte, Gamma, InputPort, Int, Load, Neq, NodeExt, OutputPort, Rvsdg, Store, Theta,
    },
    ir::Const,
    passes::{
        utils::{Changes, ConstantStore},
        Pass,
    },
    values::{Cell, Ptr},
};

use super::utils::ChangeReport;

pub struct MoveCell {
    changes: Changes<1>,
    constants: ConstantStore,
}

impl MoveCell {
    pub fn new(tape_len: u16) -> Self {
        Self {
            changes: Changes::new(["move-cell"]),
            constants: ConstantStore::new(tape_len),
        }
    }

    /// If we detect a move cell motif, this should be the generated code:
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
    fn theta_move_cell(&mut self, graph: &mut Rvsdg, theta: &mut Theta) -> bool {
        if let Some((src_ptr, dest_ptr)) = self.move_cell_inner(theta) {
            let src_ptr = match src_ptr {
                Ok(input) => graph.input_source(input),
                Err(constant) => graph.constant(constant).value(),
            };
            let dest_ptr = match dest_ptr {
                Ok(input) => graph.input_source(input),
                Err(constant) => graph.constant(constant).value(),
            };

            let input_effect = graph.input_source(theta.input_effect().unwrap());
            let zero = graph.byte(0).value();

            // src_val := load src_ptr
            let src_val = graph.load(src_ptr, input_effect);

            // store src_ptr, byte 0
            let store_src = graph.store(src_ptr, zero, src_val.output_effect());

            // store dest_ptr, src_val
            let store_dest =
                graph.store(dest_ptr, src_val.output_value(), store_src.output_effect());

            graph.rewire_dependents(theta.output_effect().unwrap(), store_dest.output_effect());
            graph.remove_node(theta.node());
            self.changes.inc::<"move-cell">();

            true
        } else {
            false
        }
    }

    // ```
    // do {
    //     // Decrement the source cell
    //     src_val := load src_ptr
    //     src_dec := sub src_val, byte 1
    //     store src_ptr, src_dec
    //
    //     // Increment the destination cell
    //     dest_val := load dest_ptr
    //     dest_inc := add dest_val, byte 1
    //     store dest_ptr, dest_inc
    //
    //     // This load is optional, the comparison can also directly
    //     // reference `src_dec`
    //     src_dec_reload := load src_ptr
    //
    //     src_is_zero := cmp.neq {src_dec, src_dec_reload}, int 0
    // } while { src_is_zero }
    // ```
    fn move_cell_inner(
        &self,
        theta: &Theta,
    ) -> Option<(Result<InputPort, Const>, Result<InputPort, Const>)> {
        let graph = theta.body();
        let start = theta.start_node();

        if theta.outputs_len() != 0 || theta.variant_inputs_len() != 0 {
            return None;
        }

        let first = self.load_mutate_store(graph, start.effect())?;

        match first.loaded_value {
            // dest_val := load dest_ptr
            // dest_inc := add dest_val, byte 1
            // store dest_ptr, dest_inc
            AddOrSub::Add(_dest_inc) => {
                // src_val := load src_ptr
                // src_dec := sub src_val, byte 1
                // store src_ptr, src_dec
                let src_dec = self.load_mutate_store(graph, first.output_effect)?;
                src_dec.loaded_value.as_sub()?;
                if first.mutated_value != src_dec.mutated_value {
                    return None;
                }

                let src_dec_port =
                    // src_dec_reload := load src_ptr
                    if let Some(reload) = graph.cast_output_dest::<Load>(src_dec.output_effect) {
                        let ptr_src = graph.input_source(reload.ptr());
                        if ptr_src != src_dec.ptr && self.constants.get(ptr_src).zip_with(self.constants.get(src_dec.ptr), |lhs, rhs| lhs != rhs).unwrap_or(true) {
                            return None;
                        } else {
                            reload.output_value()
                        }
                    } else {
                        src_dec.loaded_value.value()
                    };

                // src_is_zero := cmp.neq {src_dec, src_dec_reload}, int 0
                let src_is_zero = graph.cast_input_source::<Neq>(theta.condition().input())?;
                if graph.input_source(src_is_zero.lhs()) != src_dec_port
                    || !self
                        .constants
                        .get(graph.input_source(src_is_zero.rhs()))?
                        .is_zero()
                {
                    return None;
                }

                Some((
                    self.const_or_input(src_dec.ptr, theta)?,
                    self.const_or_input(first.ptr, theta)?,
                ))
            }

            // src_val := load src_ptr
            // src_dec := sub src_val, byte 1
            // store src_ptr, src_dec
            AddOrSub::Sub(src_dec) => {
                // dest_val := load dest_ptr
                // dest_inc := add dest_val, byte 1
                // store dest_ptr, dest_inc
                let dest_inc = self.load_mutate_store(graph, first.output_effect)?;
                dest_inc.loaded_value.as_add()?;
                if first.mutated_value != dest_inc.mutated_value {
                    return None;
                }

                let src_dec =
                    // src_dec_reload := load src_ptr
                    if let Some(reload) = graph.cast_output_dest::<Load>(dest_inc.output_effect) {
                        let ptr_src = graph.input_source(reload.ptr());
                        if ptr_src != first.ptr && self.constants.get(ptr_src).zip_with(self.constants.get(first.ptr), |lhs, rhs| lhs != rhs).unwrap_or(true) {
                            return None;
                        } else {
                            reload.output_value()
                        }
                    } else {
                       src_dec.value()
                    };

                // src_is_zero := cmp.neq {src_dec, src_dec_reload}, int 0
                let src_is_zero = graph.cast_input_source::<Neq>(theta.condition().input())?;
                if graph.input_source(src_is_zero.lhs()) != src_dec
                    || !self
                        .constants
                        .get(graph.input_source(src_is_zero.rhs()))?
                        .is_zero()
                {
                    return None;
                }

                Some((
                    self.const_or_input(first.ptr, theta)?,
                    self.const_or_input(dest_inc.ptr, theta)?,
                ))
            }
        }
    }

    fn load_mutate_store(
        &self,
        graph: &Rvsdg,
        input_effect: OutputPort,
    ) -> Option<LoadMutateStore> {
        // src_val := load src_ptr
        let src_val = graph.cast_output_dest::<Load>(input_effect)?;
        let src_ptr = graph.input_source(src_val.ptr());

        // store src_ptr, src_dec
        let src_dec_store = graph.cast_output_dest::<Store>(src_val.output_effect())?;
        if graph.input_source(src_dec_store.ptr()) != src_ptr {
            return None;
        }

        // src_dec := sub src_val, byte 1
        let src_dec = AddOrSub::cast_input_source(graph, src_dec_store.value())?;
        let src_dec_val = self.constants.get(graph.input_source(src_dec.rhs()))?;
        if graph.input_source(src_dec.lhs()) != src_val.output_value() || src_dec_val.is_zero() {
            return None;
        }

        Some(LoadMutateStore {
            output_effect: src_dec_store.output_effect(),
            loaded_value: src_dec,
            mutated_value: src_dec_val,
            ptr: src_ptr,
        })
    }

    fn const_or_input(&self, ptr: OutputPort, theta: &Theta) -> Option<Result<InputPort, Const>> {
        if let Some(value) = self.constants.get(ptr) {
            Some(Err(value))
        } else {
            let ptr_node = theta.body().port_parent(ptr);

            if let Some(input) = theta
                .invariant_input_pairs()
                .find_map(|(input, param)| (param.node() == ptr_node).then_some(input))
            {
                Some(Ok(input))
            } else {
                tracing::warn!("found move cell motif with an input pointer that was neither const nor an input param");
                None
            }
        }
    }
}

impl Pass for MoveCell {
    fn pass_name(&self) -> &'static str {
        "move-cell"
    }

    fn did_change(&self) -> bool {
        self.changes.did_change()
    }

    fn report(&self) -> ChangeReport {
        self.changes.as_report()
    }

    fn reset(&mut self) {
        self.changes.reset();
        self.constants.clear();
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        // TODO: Propagate constants
        if self.visit_graph(gamma.true_mut()) | self.visit_graph(gamma.false_mut()) {
            graph.replace_node(gamma.node(), gamma);
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        // TODO: Propagate constants

        if self.visit_graph(theta.body_mut()) & !self.theta_move_cell(graph, &mut theta) {
            graph.replace_node(theta.node(), theta);
        }
    }

    fn visit_byte(&mut self, _graph: &mut Rvsdg, byte: Byte, value: Cell) {
        self.constants.add(byte.value(), value);
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: Ptr) {
        self.constants.add(int.value(), value);
    }
}

#[derive(Debug)]
struct LoadMutateStore {
    output_effect: OutputPort,
    loaded_value: AddOrSub,
    mutated_value: Const,
    ptr: OutputPort,
}
