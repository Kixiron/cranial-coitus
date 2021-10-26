use crate::{
    graph::{Add, End, Eq, Gamma, Int, Load, Node, NodeExt, Not, OutputPort, Rvsdg, Store, Theta},
    ir::Const,
    passes::Pass,
};
use std::collections::BTreeMap;

#[derive(Debug)]
enum Candidate {
    Add,
    Sub,
}

pub struct AddSubLoop {
    changed: bool,
    values: BTreeMap<OutputPort, Const>,
}

impl AddSubLoop {
    pub fn new() -> Self {
        Self {
            values: BTreeMap::new(),
            changed: false,
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
    /// do {
    ///   // Decrement the counter
    ///   _counter_val := load _counter_ptr
    ///   _dec_counter := add _counter_val, int -1
    ///   store _counter_ptr, _counter_lhs
    ///
    ///   // Increment the accumulator
    ///   _acc_val := load _acc_ptr
    ///   _inc_acc := add _acc_val, int 1
    ///   store _acc_ptr, _inc_acc
    ///
    ///   // Compare the counter's value to zero
    ///   _counter_is_zero := eq _dec_counter, int 0
    ///   _counter_not_zero := not _counter_is_zero
    /// } while { _counter_not_zero }
    /// ```
    ///
    /// Subtraction:
    ///
    /// ```
    /// do {
    ///   // Decrement the counter
    ///   _counter_val := load _counter_ptr
    ///   _dec_counter := add _counter_val, int -1
    ///   store _counter_ptr, _counter_lhs
    ///
    ///   // Decrement the accumulator
    ///   _acc_val := load _acc_ptr
    ///   _inc_dec := add _acc_val, int -1
    ///   store _acc_ptr, _inc_dec
    ///
    ///   // Compare the counter's value to zero
    ///   _counter_is_zero := eq _dec_counter, int 0
    ///   _counter_not_zero := not _counter_is_zero
    /// } while { _counter_not_zero }
    /// ```
    ///
    /// After conversion, these should be like this:
    /// ```
    /// _counter_val := load _counter_ptr
    /// _acc_val := load _acc_ptr
    ///
    /// // Sum
    /// _sum := add _counter_val, _acc_val
    ///
    /// // Difference
    /// _neg_acc = neg _counter_val
    /// _diff := sub  _counter_val, _neg_acc
    ///
    /// // Store the sum or difference to the accumulator cell
    /// store _acc_ptr, {_sum, _diff}
    ///
    /// // Zero out the counter cell
    /// store _counter_ptr, 0
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
                .and_then(|val| val.convert_to_i32())
                .map_or(false, |val| val == expected)
        };

        let start = theta.start_node();

        // The initial load from the lhs ptr
        // _counter_val := load _counter_ptr
        let load_counter = graph.cast_target::<Load>(start.effect())?;
        let counter_ptr = graph.input_source(load_counter.ptr());

        // Stores the sum/diff to the lhs ptr
        // store _counter_ptr, _counter_lhs
        let store_dec_counter = graph.cast_target::<Store>(load_counter.effect())?;

        // If they store to different places this isn't a candidate
        if graph.input_source(store_dec_counter.ptr()) != counter_ptr {
            return None;
        }

        // Decrements the lhs cell's value
        // _dec_counter := add _counter_val, int -1
        let dec_counter = graph.cast_source::<Add>(store_dec_counter.value())?;

        let (add_lhs_src, add_rhs_src) = (
            graph.get_input(dec_counter.lhs()).1,
            graph.get_input(dec_counter.rhs()).1,
        );
        let (add_lhs_neg_one, add_rhs_neg_one) = (is(add_lhs_src, -1), is(add_rhs_src, -1));

        // If the add isn't `add _loaded_lhs, -1` this isn't a candidate
        if !((add_lhs_src == load_counter.value() && add_rhs_neg_one)
            || (add_lhs_neg_one && add_rhs_src == load_counter.value()))
        {
            return None;
        }

        // The load of the rhs
        // _rhs_acc := load _acc_ptr
        let load_acc = graph.cast_target::<Load>(store_dec_counter.effect())?;
        let acc_ptr = graph.input_source(load_acc.ptr());

        // Stores the decremented or incremented value to the rhs cell
        // store _acc_ptr, {_inc_acc, _dec_acc}
        let store_inc_dec_acc = graph.cast_target::<Store>(load_acc.effect())?;

        // If the pointer isn't the rhs pointer, this isn't a candidate
        if graph.input_source(store_inc_dec_acc.ptr()) != acc_ptr {
            return None;
        }

        // Ensure that the store is the last effect within the loop
        let _end = graph.cast_target::<End>(store_inc_dec_acc.effect())?;

        // Either incrementing or decrementing the
        // _inc_acc := add _acc_val, int 1
        // _inc_dec := add _acc_val, int -1
        let inc_dec_acc = graph.cast_source::<Add>(store_inc_dec_acc.value())?;

        let (add_lhs_src, add_rhs_src) = (
            graph.get_input(inc_dec_acc.lhs()).1,
            graph.get_input(inc_dec_acc.rhs()).1,
        );
        let (add_lhs_one, add_rhs_one) = (is(add_lhs_src, 1), is(add_rhs_src, 1));
        let (add_lhs_neg_one, add_rhs_neg_one) = (is(add_lhs_src, -1), is(add_rhs_src, -1));

        // _inc_acc := add _acc_val, int 1
        let candidate = if (add_lhs_src == load_acc.value() && add_rhs_one)
            || (add_lhs_one && add_rhs_src == load_acc.value())
        {
            Candidate::Add

        // _inc_dec := add _acc_val, int -1
        } else if (add_lhs_src == load_acc.value() && add_rhs_neg_one)
            || (add_lhs_neg_one && add_rhs_src == load_acc.value())
        {
            Candidate::Sub

        // Otherwise this isn't a candidate
        } else {
            return None;
        };

        // The output param that takes the theta's exit condition
        let cond_output = theta.condition();

        // Invert the `_dec_counter == 0` to make it `_dec_counter != 0`
        // _counter_not_zero := not _counter_is_zero
        let cond_neg = graph.cast_source::<Not>(cond_output.input())?;

        // The check if the counter's decremented value is zero
        // _counter_is_zero := eq _dec_counter, int 0
        let cond_is_zero = graph.cast_source::<Eq>(cond_neg.input())?;

        let (eq_lhs_src, eq_rhs_src) = (
            graph.get_input(cond_is_zero.lhs()).1,
            graph.get_input(cond_is_zero.rhs()).1,
        );
        let (eq_lhs_zero, eq_rhs_zero) = (is(eq_lhs_src, 0), is(eq_rhs_src, 0));

        // If the eq isn't `eq _dec_counter, 0` this isn't a candidate
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

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: i32) {
        let replaced = self.values.insert(int.value(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Int(value)));
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let (mut truthy_visitor, mut falsy_visitor) = (Self::new(), Self::new());

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[true_param, false_param]) in gamma.inputs().iter().zip(gamma.input_params())
        {
            let (_, source, _) = graph.get_input(input);

            if let Some(constant) = self.values.get(&source).cloned() {
                let true_param = gamma.true_branch().get_node(true_param).to_input_param();
                let replaced = truthy_visitor
                    .values
                    .insert(true_param.output(), constant.clone());
                debug_assert!(replaced.is_none());

                let false_param = gamma.false_branch().get_node(false_param).to_input_param();
                let replaced = falsy_visitor.values.insert(false_param.output(), constant);
                debug_assert!(replaced.is_none());
            }
        }

        truthy_visitor.visit_graph(gamma.truthy_mut());
        falsy_visitor.visit_graph(gamma.falsy_mut());
        self.changed |= truthy_visitor.did_change();
        self.changed |= falsy_visitor.did_change();

        graph.replace_node(gamma.node(), gamma);
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut visitor = Self::new();

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (input, param) in theta.input_pairs() {
            if let Some(constant) = self.values.get(&graph.input_source(input)).cloned() {
                let replaced = visitor.values.insert(param.output(), constant);
                debug_assert!(replaced.is_none());
            }
        }

        visitor.visit_graph(theta.body_mut());
        self.changed |= visitor.did_change();

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
            let acc_val = graph.load(acc_ptr, counter_val.effect());

            let sum_diff = match candidate {
                // Add the counter and accumulator values together
                Candidate::Add => graph.add(counter_val.value(), acc_val.value()).value(),

                // Negate the accumulator value and then add the negated
                // accumulator and counter values together
                Candidate::Sub => {
                    let neg_acc = graph.neg(acc_val.value());

                    graph.add(counter_val.value(), neg_acc.value()).value()
                }
            };

            // Store the sum or difference to the accumulator cell
            let store_sum_diff = graph.store(acc_ptr, sum_diff, acc_val.effect());

            // Unconditionally store 0 to the counter cell
            let zero = graph.int(0);
            let zero_counter = graph.store(counter_ptr, zero.value(), store_sum_diff.effect());

            // Wire the final store into the theta's output effect
            graph.rewire_dependents(theta.output_effect().unwrap(), zero_counter.effect());

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

                        Node::InputPort(param) => {
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
        } else {
            graph.replace_node(theta.node(), theta);
        }
    }
}

impl Default for AddSubLoop {
    fn default() -> Self {
        Self::new()
    }
}
