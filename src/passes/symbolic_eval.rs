use crate::{
    graph::{Bool, Gamma, Int, Load, NodeExt, OutputPort, PortId, Rvsdg, Store, Theta},
    interpreter::Machine,
    ir::{Const, IrBuilder, Value, VarId},
    passes::Pass,
    utils::AssertNone,
};
use std::{collections::BTreeMap, mem};

pub struct SymbolicEval {
    changed: bool,
    tape: Vec<Option<u8>>,
    values: BTreeMap<OutputPort, Const>,
}

impl SymbolicEval {
    pub fn new(tape_len: usize) -> Self {
        Self {
            changed: false,
            tape: vec![Some(0); tape_len],
            values: BTreeMap::new(),
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
            *cell = Some(0);
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
        self.values.clear();
        self.zero_tape();
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        let replaced = self.values.insert(bool.value(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Bool(value)));
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: i32) {
        let replaced = self.values.insert(int.value(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Int(value)));
    }

    fn visit_store(&mut self, graph: &mut Rvsdg, store: Store) {
        let ptr = self
            .values
            .get(&graph.input_source(store.ptr()))
            .and_then(Const::convert_to_i32)
            .map(|ptr| ptr.rem_euclid(self.tape.len() as i32) as usize);

        let value = self
            .values
            .get(&graph.input_source(store.value()))
            .and_then(Const::convert_to_u8);

        if let Some(ptr) = ptr {
            self.tape[ptr] = value;
        } else {
            self.clear_tape();
        }
    }

    fn visit_load(&mut self, graph: &mut Rvsdg, load: Load) {
        let ptr = self
            .values
            .get(&graph.input_source(load.ptr()))
            .and_then(Const::convert_to_i32)
            .map(|ptr| ptr.rem_euclid(self.tape.len() as i32) as usize);

        if let Some(value) = ptr.and_then(|ptr| self.tape[ptr]) {
            self.values
                .insert(load.output_value(), value.into())
                .debug_unwrap_none();
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, theta: Theta) {
        let tape_len = self.tape.len();

        // We don't currently accept variant inputs
        if theta.variant_inputs_len() != 0 {
            tracing::trace!(
                theta = ?theta.node(),
                "failed to evaluate theta node, it has variant inputs",
            );
            self.clear_tape();

            return;
        }

        // If all of our input values aren't const-known we can't continue
        let mut values = BTreeMap::new();
        for (port, param) in theta.variant_input_pairs() {
            let source = graph.input_source(port);

            if let Some(value) = self.values.get(&source).cloned() {
                values
                    .insert(VarId::new(param.output()), value)
                    .debug_unwrap_none();
            } else {
                tracing::trace!(
                    theta = ?theta.node(),
                    "failed to evaluate theta node, it's missing const inputs",
                );
                self.clear_tape();

                return;
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

            return;
        }

        let mut machine = Machine::new(1_000_000, tape_len, || unreachable!(), |_| unreachable!());
        // Give the machine our current tape state
        machine.tape = mem::replace(&mut self.tape, vec![None; tape_len]);

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

        let mut body = builder.translate(theta.body());

        // If we successfully evaluate the ir, we want to retain its output state
        match machine.execute(&mut body, false) {
            Ok(output_tape) => {
                tracing::trace!(theta = ?theta.node(), "evaluated theta node");
                self.tape = output_tape.to_vec();

                for (output, param) in theta.output_pairs() {
                    let source = theta.body().input_source(param.input());

                    let value = machine
                        .values
                        .last()
                        .unwrap()
                        .get(&VarId::new(source))
                        .unwrap()
                        .clone();

                    let int = graph.int(value.convert_to_i32().unwrap());
                    self.values.insert(output, value).debug_unwrap_none();
                    graph.rewire_dependents(output, int.value());
                }

                graph.remove_node(theta.node());
                self.changed();
            }

            Err(error) => {
                tracing::trace!(theta = ?theta.node(), "failed to evaluate theta node: {:?}", error);
            }
        }
    }

    // TODO: Evaluate within gamma nodes
    fn visit_gamma(&mut self, _graph: &mut Rvsdg, _gamma: Gamma) {
        self.clear_tape();
    }
}
