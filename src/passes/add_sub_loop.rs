use crate::{
    graph::{
        Add, Byte, End, Eq, Gamma, InputParam, Int, Load, Node, NodeExt, Not, OutputPort, Rvsdg,
        Store, Sub, Theta,
    },
    ir::Const,
    passes::Pass,
    utils::{AssertNone, HashMap},
    values::{Cell, Ptr},
};
use std::collections::BTreeMap;

#[derive(Debug)]
enum Candidate {
    Add,
    Sub,
}

pub struct AddSubLoop {
    changed: bool,
    // FIXME: ConstantStore
    values: BTreeMap<OutputPort, Const>,
    add_loops_removed: usize,
    sub_loops_removed: usize,
    tape_len: u16,
}

impl AddSubLoop {
    pub fn new(tape_len: u16) -> Self {
        Self {
            values: BTreeMap::new(),
            changed: false,
            add_loops_removed: 0,
            sub_loops_removed: 0,
            tape_len,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    /// We're matching two motifs here
    ///
    /// Addition:
    ///
    /// ```
    /// // TODO: Account for both orientations of the counter/accum decrementing
    /// //       and/or use `psi` nodes to make the state disjoint
    /// do {
    ///   // Decrement the counter
    ///   counter_val := load counter_ptr
    ///   dec_counter := sub counter_val, int 1
    ///   store counter_ptr, counter_lhs
    ///
    ///   // Increment the accumulator
    ///   acc_val := load acc_ptr
    ///   inc_acc := add acc_val, int 1
    ///   store acc_ptr, inc_acc
    ///
    ///   // Compare the counter's value to zero
    ///   counter_is_zero := eq dec_counter, int 0
    ///   counter_not_zero := not counter_is_zero
    /// } while { counter_not_zero }
    /// ```
    ///
    /// Subtraction:
    ///
    /// ```
    /// // TODO: Account for both orientations of the counter/accum decrementing
    /// //       and/or use `psi` nodes to make the state disjoint
    /// do {
    ///   // Decrement the counter
    ///   counter_val := load counter_ptr
    ///   dec_counter := sub counter_val, int 1
    ///   store counter_ptr, counter_lhs
    ///
    ///   // Decrement the accumulator
    ///   acc_val := load _acc_ptr
    ///   dec_acc := sub acc_val, int 1
    ///   store acc_ptr, dec_acc
    ///
    ///   // Compare the counter's value to zero
    ///   counter_is_zero := eq dec_counter, int 0
    ///   counter_not_zero := not counter_is_zero
    /// } while { counter_not_zero }
    /// ```
    ///
    /// After conversion, these should be like this:
    /// ```
    /// counter_val := load counter_ptr
    /// acc_val := load acc_ptr
    ///
    /// // Sum
    /// sum := add counter_val, acc_val
    ///
    /// // Difference
    /// diff := sub counter_val, neg_acc
    ///
    /// // Store the sum or difference to the accumulator cell
    /// store acc_ptr, {sum, diff}
    ///
    /// // Zero out the counter cell
    /// store counter_ptr, 0
    /// ```
    // FIXME: Technically we need to match 4 motifs: These two can be
    //        reversed to turn `[->+<]` (add) and `[-<->]` (sub) into
    //        `[>+<-]` (add) and `[<->-]` (sub).
    //        Wait, shouldn't the RVSDG normalize the two pairs into
    //        each other so that detecting just the one will work?
    fn theta_is_candidate(
        &self,
        values: &BTreeMap<OutputPort, Const>,
        theta: &Theta,
    ) -> Option<(Candidate, OutputPort, OutputPort)> {
        let graph = theta.body();
        let is = |src, expected| {
            values
                .get(&src)
                .map_or(false, |value| value.into_ptr(self.tape_len) == expected)
        };

        let start = theta.start_node();

        // The initial load from the lhs ptr
        // counter_val := load counter_ptr
        let load_counter = graph.cast_target::<Load>(start.effect())?;
        let counter_ptr = graph.input_source(load_counter.ptr());

        // Stores the sum/diff to the lhs ptr
        // store counter_ptr, counter_lhs
        let store_dec_counter = graph.cast_target::<Store>(load_counter.output_effect())?;

        // If they store to different places this isn't a candidate
        if graph.input_source(store_dec_counter.ptr()) != counter_ptr {
            return None;
        }

        // Decrements the lhs cell's value
        // dec_counter := sub counter_val, int 1
        let dec_counter = graph.cast_source::<Sub>(store_dec_counter.value())?;

        let (dec_lhs_src, add_rhs_src) = (
            graph.input_source(dec_counter.lhs()),
            graph.input_source(dec_counter.rhs()),
        );
        let (add_lhs_neg_one, add_rhs_neg_one) = (is(dec_lhs_src, 1), is(add_rhs_src, 1));

        // If the sub isn't `sub loaded_lhs, 1` this isn't a candidate
        if !((dec_lhs_src == load_counter.output_value() && add_rhs_neg_one)
            || (add_lhs_neg_one && add_rhs_src == load_counter.output_value()))
        {
            return None;
        }

        // The load of the rhs
        // rhs_acc := load acc_ptr
        let load_acc = graph.cast_target::<Load>(store_dec_counter.output_effect())?;
        let acc_ptr = graph.input_source(load_acc.ptr());

        // Stores the decremented or incremented value to the rhs cell
        // store acc_ptr, {inc_acc, dec_acc}
        let store_inc_dec_acc = graph.cast_target::<Store>(load_acc.output_effect())?;

        // If the pointer isn't the rhs pointer, this isn't a candidate
        if graph.input_source(store_inc_dec_acc.ptr()) != acc_ptr {
            return None;
        }

        // Ensure that the store is the last effect within the loop
        let _end = graph.cast_target::<End>(store_inc_dec_acc.output_effect())?;

        // Either incrementing or decrementing the
        // inc_acc := add acc_val, int 1
        // dec_acc := sub acc_val, int 1
        let candidate = if let Some(add) = graph.cast_source::<Add>(store_inc_dec_acc.value()) {
            let (lhs, rhs) = (graph.input_source(add.lhs()), graph.input_source(add.rhs()));
            let (lhs_one, rhs_one) = (is(lhs, 1), is(rhs, 1));

            // inc_acc := add acc_val, int 1
            if (lhs == load_acc.output_value() && rhs_one)
                || (lhs_one && rhs == load_acc.output_value())
            {
                Candidate::Add
            } else {
                return None;
            }
        } else if let Some(sub) = graph.cast_source::<Sub>(store_inc_dec_acc.value()) {
            let (lhs, rhs) = (graph.input_source(sub.lhs()), graph.input_source(sub.rhs()));
            let (lhs_one, rhs_one) = (is(lhs, 1), is(rhs, 1));

            // dec_acc := sub acc_val, int 1
            if (lhs == load_acc.output_value() && rhs_one)
                || (lhs_one && rhs == load_acc.output_value())
            {
                Candidate::Sub
            } else {
                return None;
            }
        } else {
            return None;
        };

        // The output param that takes the theta's exit condition
        let cond_output = theta.condition();

        // Invert the `dec_counter == 0` to make it `dec_counter != 0`
        // counter_not_zero := not counter_is_zero
        let cond_neg = graph.cast_source::<Not>(cond_output.input())?;

        // The check if the counter's decremented value is zero
        // counter_is_zero := eq dec_counter, int 0
        let cond_is_zero = graph.cast_source::<Eq>(cond_neg.input())?;

        let (eq_lhs_src, eq_rhs_src) = (
            graph.get_input(cond_is_zero.lhs()).1,
            graph.get_input(cond_is_zero.rhs()).1,
        );
        let (eq_lhs_zero, eq_rhs_zero) = (is(eq_lhs_src, 0), is(eq_rhs_src, 0));

        // If the eq isn't `eq dec_counter, 0` this isn't a candidate
        if !((eq_lhs_src == dec_counter.value() && eq_rhs_zero)
            || (eq_lhs_zero && eq_rhs_src == dec_counter.value()))
        {
            return None;
        }

        // Otherwise we've matched the pattern properly and this is an add/sub loop
        Some((candidate, counter_ptr, acc_ptr))
    }
}

impl Pass for AddSubLoop {
    fn pass_name(&self) -> &str {
        "add-sub-loop"
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
            "addition loops" => self.add_loops_removed,
            "subtraction loops" => self.sub_loops_removed,
        }
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: Ptr) {
        self.values.insert(int.value(), value.into());
    }

    fn visit_byte(&mut self, _graph: &mut Rvsdg, byte: Byte, value: Cell) {
        self.values.insert(byte.value(), value.into());
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;
        let (mut true_visitor, mut false_visitor) =
            (Self::new(self.tape_len), Self::new(self.tape_len));

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

        if changed {
            graph.replace_node(gamma.node(), gamma);
            self.changed();
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut changed = false;
        let mut visitor = Self::new(self.tape_len);

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

        if let Some((candidate, counter_ptr, acc_ptr)) =
            self.theta_is_candidate(&visitor.values, &theta)
        {
            tracing::debug!(
                "found theta that composes to a{} loop of {:?} {} {:?}",
                match candidate {
                    Candidate::Add => "n add",
                    Candidate::Sub => " sub",
                },
                counter_ptr,
                match candidate {
                    Candidate::Add => "+",
                    Candidate::Sub => "-",
                },
                acc_ptr,
            );

            let mut get_theta_input = |output| {
                theta
                    .input_pairs()
                    .find_map(|(port, input)| {
                        if input.output() == output {
                            Some(graph.get_input(port).1)
                        } else {
                            None
                        }
                    })
                    .or_else(|| {
                        theta
                            .body()
                            .get_node(theta.body().port_parent(output))
                            .as_int()
                            .map(|(_, int)| graph.int(int).value())
                    })
                    .unwrap()
            };

            let (counter_ptr, acc_ptr) = (get_theta_input(counter_ptr), get_theta_input(acc_ptr));

            let input_effect = graph.input_source(theta.input_effect().unwrap());

            // Load the counter and accumulator values
            let counter_val = graph.load(counter_ptr, input_effect);
            let acc_val = graph.load(acc_ptr, counter_val.output_effect());

            let sum_diff = match candidate {
                // Add the counter and accumulator values together
                Candidate::Add => {
                    self.add_loops_removed += 1;
                    graph
                        .add(counter_val.output_value(), acc_val.output_value())
                        .value()
                }

                // Subtract the accumulator from the counter
                Candidate::Sub => {
                    self.sub_loops_removed += 1;
                    graph
                        .sub(counter_val.output_value(), acc_val.output_value())
                        .value()
                }
            };

            // Store the sum or difference to the accumulator cell
            let store_sum_diff = graph.store(acc_ptr, sum_diff, acc_val.output_effect());

            // Unconditionally store 0 to the counter cell
            let zero = graph.int(Ptr::zero(self.tape_len));
            let zero_counter =
                graph.store(counter_ptr, zero.value(), store_sum_diff.output_effect());

            // Wire the final store into the theta's output effect
            graph.rewire_dependents(theta.output_effect().unwrap(), zero_counter.output_effect());

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
        } else if changed {
            graph.replace_node(theta.node(), theta);
            self.changed();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::passes::{
        AddSubLoop, AssociativeOps, ConstFolding, ElimConstGamma, ExprDedup, Mem2Reg,
        UnobservedStore, ZeroLoop,
    };

    test_opts! {
        // ```bf
        // y[-x+y]x
        // ```
        addition_motif_one,
        passes = |tape_len| -> Vec<Box<dyn Pass + 'static>> {
            bvec![
                Dce::new(),
                UnobservedStore::new(tape_len),
                ConstFolding::new(tape_len),
                AssociativeOps::new(tape_len),
                ZeroLoop::new(tape_len),
                Mem2Reg::new(tape_len),
                AddSubLoop::new(tape_len),
                Dce::new(),
                ElimConstGamma::new(),
                ConstFolding::new(tape_len),
                ExprDedup::new(),
            ]
        },
        input = [10, 20],
        output = [0, 30],
        |graph, mut effect, tape_len| {
            let y_ptr = graph.int(Ptr::zero(tape_len)).value();
            let x_ptr = graph.int(Ptr::one(tape_len)).value();

            // Get and store the y value
            let y_input = graph.input(effect);
            effect = y_input.output_effect();
            let store = graph.store(y_ptr, y_input.output_value(), effect);
            effect = store.output_effect();

            // Get and store the x value
            let x_input = graph.input(effect);
            effect = x_input.output_effect();
            let store = graph.store(x_ptr, x_input.output_value(), effect);
            effect = store.output_effect();

            // Compile the loop
            let (x_ptr, mut effect) = compile_brainfuck_into("[->+<]>", graph, y_ptr, effect, tape_len);

            // Print the y value
            let y_value = graph.load(y_ptr, effect);
            effect = y_value.output_effect();
            let output = graph.output(y_value.output_value(), effect);
            effect = output.output_effect();

            // Print the x value
            let x_value = graph.load(x_ptr, effect);
            effect = x_value.output_effect();
            let output = graph.output(x_value.output_value(), effect);
            effect = output.output_effect();

            effect
        },
    }

    // TODO: Detect & optimize this motif (post zero-loop optimization)
    // ```bf
    // temp0[-]
    // y[x+temp0+y-]
    // temp0[y+temp0-]
    // ```
}
