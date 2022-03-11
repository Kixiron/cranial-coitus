use crate::{
    graph::{Bool, Byte, Gamma, Int, Load, NodeExt, OutputPort, PortId, Rvsdg, Store, Theta},
    interpreter::Machine,
    ir::{IrBuilder, Pretty, PrettyConfig, Value, VarId},
    passes::{utils::ConstantStore, Pass},
    utils::{self, AssertNone, HashMap},
    values::{Cell, Ptr},
};
use std::collections::BTreeMap;

pub struct SymbolicEval {
    changed: bool,
    tape: Vec<Option<Cell>>,
    tape_len: u16,
    constants: ConstantStore,
    evaluated_outputs: usize,
    evaluated_thetas: usize,
}

impl SymbolicEval {
    pub fn new(tape_len: u16) -> Self {
        Self {
            changed: false,
            tape: vec![Some(Cell::zero()); tape_len as usize],
            constants: ConstantStore::new(tape_len),
            evaluated_outputs: 0,
            evaluated_thetas: 0,
            tape_len,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    fn clear_tape(&mut self) {
        for cell in &mut self.tape {
            *cell = None;
        }
    }

    fn zero_tape(&mut self) {
        for cell in &mut self.tape {
            *cell = Some(Cell::zero());
        }
    }

    fn try_evaluate_theta(&mut self, graph: &mut Rvsdg, theta: &Theta) -> Option<OutputPort> {
        // We don't currently accept variant inputs
        // TODO: .has_variant_inputs()
        if theta.variant_inputs_len() != 0 {
            tracing::trace!(
                theta = ?theta.node(),
                "failed to evaluate theta node, it has variant inputs",
            );
            self.clear_tape();

            return None;
        }

        // If all of our input values aren't const-known we can't continue
        let (mut values, mut inputs) = (BTreeMap::new(), BTreeMap::new());
        for (port, param) in theta.invariant_input_pairs() {
            let source = graph.input_source(port);

            if let Some(value) = self.constants.get(source) {
                values
                    .insert(VarId::new(param.output()), value)
                    .debug_unwrap_none();

                inputs.insert(port, Value::Const(value)).debug_unwrap_none();
            } else {
                tracing::trace!(
                    theta = ?theta.node(),
                    "failed to evaluate theta node, it's missing const inputs",
                );
                self.clear_tape();

                return None;
            }
        }

        // We don't want io operations or gamma/theta nodes
        // Gamma and theta nodes could be evaluated, but I don't have the capacity
        // for that right now
        let mut contains_disallowed_nodes = false;
        theta.body().for_each_transitive_node(|_, node| {
            contains_disallowed_nodes |=
                node.is_input() || node.is_output() || node.is_gamma() || node.is_theta();
        });
        if contains_disallowed_nodes {
            tracing::trace!(
                theta = ?theta.node(),
                "failed to evaluate theta node, it contains disallowed nodes",
            );
            self.clear_tape();

            return None;
        }

        let mut machine = Machine::new(
            100_000_000,
            self.tape_len,
            || unreachable!(),
            |_| unreachable!(),
        );
        // Give the machine our current tape state
        machine.tape = self.tape.clone();

        // Add all the input values to the machine
        machine
            .values
            .last_mut()
            .unwrap()
            .append(&mut values.clone());
        machine.values.insert(0, values.clone());
        machine.values_idx += 1;

        // Translate the theta's body into ir
        let mut builder = IrBuilder::new(false);
        let mut values: BTreeMap<_, _> = values
            .into_iter()
            .map(|(var, val)| (OutputPort::new(PortId::new(var.0)), Value::Const(val)))
            .collect();
        builder.values.append(&mut values);

        let mut builder = IrBuilder::new(false);
        builder.push_theta(graph, &inputs, theta);

        let mut body = builder.finish();
        tracing::debug!(
            theta = ?theta.node(),
            "symbolically evaluating theta node: {}",
            body.pretty_print(PrettyConfig::minimal()),
        );

        // If we successfully evaluate the ir, we want to retain its output state
        match machine
            .execute(&mut body, false)
            .map(|output_tape| output_tape.to_vec())
        {
            Ok(output_tape) => {
                tracing::debug!(
                    theta = ?theta.node(),
                    input_tape = ?utils::debug_collapse(&self.tape),
                    output_tape = ?utils::debug_collapse(&output_tape),
                    "successfully evaluated theta node",
                );

                for (output, param) in theta.output_pairs() {
                    let source = theta.body().input_source(param.input());

                    let value = machine
                        .values
                        .last()
                        .unwrap()
                        .get(&VarId::new(source))
                        .unwrap()
                        .into_ptr(self.tape_len);

                    let int = graph.int(value);
                    self.constants.add(output, value);
                    graph.rewire_dependents(output, int.value());

                    self.evaluated_outputs += 1;
                }

                let input_effect = theta.input_effect().unwrap();
                let mut last_effect = graph.input_source(input_effect);

                // FIXME: I have no idea what this is doing anymore
                let mut created_values = BTreeMap::new();
                for (cell, (old, new)) in self
                    .tape
                    .iter()
                    .copied()
                    .zip(output_tape.iter().copied())
                    .enumerate()
                {
                    if let (Some(old), Some(new)) = (old, new) {
                        if old != new {
                            let ptr = *created_values.entry(cell as u16).or_insert_with(|| {
                                graph.int(Ptr::new(cell as u16, self.tape_len)).value()
                            });
                            let value = *created_values
                                .entry(new.into_inner() as u16)
                                .or_insert_with(|| graph.int(new.into_ptr(self.tape_len)).value());

                            let store = graph.store(ptr, value, last_effect);
                            last_effect = store.output_effect();
                        }
                    } else if let Some(value) = new {
                        let ptr = *created_values.entry(cell as u16).or_insert_with(|| {
                            graph.int(Ptr::new(cell as u16, self.tape_len)).value()
                        });
                        let value = *created_values
                            .entry(value.into_inner() as u16)
                            .or_insert_with(|| graph.int(value.into_ptr(self.tape_len)).value());

                        let store = graph.store(ptr, value, last_effect);
                        last_effect = store.output_effect();
                    }
                }

                self.tape = output_tape;
                Some(last_effect)
            }

            Err(error) => {
                tracing::trace!(
                    theta = ?theta.node(),
                    "failed to evaluate theta node: {:?}",
                    error,
                );

                None
            }
        }
    }
}

impl Pass for SymbolicEval {
    fn pass_name(&self) -> &str {
        "symbolic-eval"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.changed = false;
        self.constants.clear();
        self.zero_tape();
    }

    fn report(&self) -> HashMap<&'static str, usize> {
        map! {
            "evaluated outputs" => self.evaluated_outputs,
            "evaluated thetas" => self.evaluated_thetas,
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

    fn visit_store(&mut self, graph: &mut Rvsdg, store: Store) {
        if let Some(ptr) = self.constants.ptr(graph.input_source(store.ptr())) {
            self.tape[ptr] = self
                .constants
                .cell(graph.input_source(store.value()))
                .map(Cell::into);
        } else {
            self.clear_tape();
        }
    }

    fn visit_load(&mut self, graph: &mut Rvsdg, load: Load) {
        let ptr = self.constants.ptr(graph.input_source(load.ptr()));

        if let Some(value) = ptr.and_then(|ptr| self.tape[ptr]) {
            // let int = graph.int(value as i32);

            self.constants.add(load.output_value(), value);
            // self.constants.add(int.value(), value);
            //
            // graph.rewire_dependents(load.output_value(), int.value());
            // graph.splice_ports(load.input_effect(), load.output_effect());
            // graph.remove_node(load.node());
            //
            // self.removed_loads += 1;
            // self.changed();
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        if let Some(last_effect) = self.try_evaluate_theta(graph, &theta) {
            let output_effect = theta.output_effect().unwrap();

            graph.rewire_dependents(output_effect, last_effect);
            graph.remove_node(theta.node());

            self.evaluated_thetas += 1;
            self.changed();
        } else {
            let mut body_has_stores = false;
            theta
                .body()
                .for_each_transitive_node(|_node_id, node| body_has_stores |= node.is_store());

            let mut changed = false;
            let mut visitor = Self::new(0);
            visitor.tape_len = self.tape_len;

            visitor.tape = self.tape.clone();
            self.constants
                .theta_invariant_inputs_into(&theta, graph, &mut visitor.constants);

            changed |= visitor.visit_graph(theta.body_mut());

            if changed {
                graph.replace_node(theta.node(), theta);
                self.changed();
            }

            if body_has_stores {
                self.clear_tape();
            }
        }
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut has_stores = false;
        gamma
            .true_branch()
            .for_each_transitive_node(|_node_id, node| has_stores |= node.is_store());
        gamma
            .false_branch()
            .for_each_transitive_node(|_node_id, node| has_stores |= node.is_store());

        let mut changed = false;
        let (mut true_visitor, mut false_visitor) = (Self::new(0), Self::new(0));
        true_visitor.tape_len = self.tape_len;
        false_visitor.tape_len = self.tape_len;

        true_visitor.tape = self.tape.clone();
        false_visitor.tape = self.tape.clone();
        self.constants.gamma_inputs_into(
            &gamma,
            graph,
            &mut true_visitor.constants,
            &mut false_visitor.constants,
        );

        changed |= true_visitor.visit_graph(gamma.true_mut());
        changed |= false_visitor.visit_graph(gamma.false_mut());

        if changed {
            graph.replace_node(gamma.node(), gamma);
            self.changed();
        }

        if has_stores {
            self.clear_tape();
        }
    }
}
