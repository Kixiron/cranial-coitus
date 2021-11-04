use crate::{
    graph::{
        Add, Bool, Eq, Gamma, InputParam, InputPort, Int, Load, NodeExt, OutputPort, Rvsdg, Store,
        Theta,
    },
    ir::Const,
    passes::Pass,
    utils::{AssertNone, HashMap},
};

/// Removes dead code from the graph
pub struct Dataflow {
    changed: bool,
    facts: HashMap<OutputPort, Facts>,
    constants: HashMap<OutputPort, Const>,
    tape: Vec<(Option<Facts>, Option<Const>)>,
}

#[derive(Debug, Clone, Default)]
pub struct Facts {
    pub is_zero: Option<bool>,
    pub parity: Option<Parity>,
    pub sign: Option<Sign>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parity {
    Even,
    Odd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    Positive,
    Negative,
}

impl Dataflow {
    pub fn new(cells: usize) -> Self {
        Self {
            changed: false,
            facts: HashMap::with_hasher(Default::default()),
            constants: HashMap::with_hasher(Default::default()),
            tape: vec![(None, None); cells],
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    /// Returns `true` if `port` is zero. `false` means that the zero-ness
    /// of `port` is unspecified
    fn is_zero(&self, port: OutputPort) -> bool {
        self.facts
            .get(&port)
            .and_then(|facts| facts.is_zero)
            .unwrap_or(false)
    }

    /// Returns `true` if `port` is zero. `false` means that the zero-ness
    /// of `port` is unspecified
    fn source_is_zero(&self, graph: &Rvsdg, port: InputPort) -> bool {
        self.facts
            .get(&graph.input_source(port))
            .and_then(|facts| facts.is_zero)
            .unwrap_or(false)
    }
}

// Elide branches where variables are known to be non-zero
// ```
// v330 := add v324, v2401   // invocations: 101 (0.01%)
// v333 := load v330         // eff: e334, pred: e323, loads: 101 (0.01%)
// v337 := eq v333, v2407    // invocations: 101 (0.01%)
// // node: n131, eff: e416, pred: e334, branches: 101, false branches: 101
// if v337 { .. } else { .. }
// ```
impl Pass for Dataflow {
    fn pass_name(&self) -> &str {
        "dataflow"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.changed = false;
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;
        let (mut true_visitor, mut false_visitor) = (
            Self {
                tape: self.tape.clone(),
                ..Self::new(self.tape.len())
            },
            Self {
                tape: self.tape.clone(),
                ..Self::new(self.tape.len())
            },
        );

        for (&input, &[true_param, false_param]) in gamma.inputs().iter().zip(gamma.input_params())
        {
            let source = graph.input_source(input);
            let (true_param, false_param) = (
                gamma.true_branch().to_node::<InputParam>(true_param),
                gamma.false_branch().to_node::<InputParam>(false_param),
            );

            // Propagate facts
            if let Some(facts) = self.facts.get(&source).cloned() {
                true_visitor
                    .facts
                    .insert(true_param.output(), facts.clone())
                    .debug_unwrap_none();

                false_visitor
                    .facts
                    .insert(false_param.output(), facts)
                    .debug_unwrap_none();
            }

            // Propagate constants
            if let Some(constant) = self.constants.get(&source).cloned() {
                true_visitor
                    .constants
                    .insert(true_param.output(), constant.clone())
                    .debug_unwrap_none();

                false_visitor
                    .constants
                    .insert(false_param.output(), constant)
                    .debug_unwrap_none();
            }
        }

        let mut zero_fact_true_branch = |source, is_zero| {
            tracing::trace!(
                "inferred that {:?} is{} zero within {:?}'s true branch",
                source,
                if is_zero { "" } else { "n't" },
                gamma.node(),
            );

            true_visitor
                .facts
                .entry(source)
                .and_modify(|facts| facts.is_zero = Some(is_zero))
                .or_insert_with(|| Facts {
                    is_zero: Some(is_zero),
                    ..Default::default()
                });
        };
        let mut zero_fact_false_branch = |source, is_zero| {
            tracing::trace!(
                "inferred that {:?} is{} zero within {:?}'s false branch",
                source,
                if is_zero { "" } else { "n't" },
                gamma.node(),
            );

            false_visitor
                .facts
                .entry(source)
                .and_modify(|facts| facts.is_zero = Some(is_zero))
                .or_insert_with(|| Facts {
                    is_zero: Some(is_zero),
                    ..Default::default()
                });
        };

        let condition = graph.input_source_node(gamma.condition());

        // if !(x == y) { ... } else { ... }
        if let Some(not) = condition.as_not() {
            if let Some(eq) = graph.input_source_node(not.input()).as_eq() {
                let ((lhs_node, lhs_src, _), (rhs_node, rhs_src, _)) =
                    (graph.get_input(eq.lhs()), graph.get_input(eq.rhs()));

                // If the left hand operand is zero
                if self.is_zero(lhs_src) || lhs_node.as_int().map_or(false, |(_, value)| value == 0)
                {
                    // The left hand side is zero in both branches
                    zero_fact_true_branch(lhs_src, true);
                    zero_fact_false_branch(lhs_src, true);

                    // The right hand side is zero in the false branch and non-zero in the true branch
                    zero_fact_true_branch(rhs_src, false);
                    zero_fact_false_branch(rhs_src, true);

                // If the right hand operand is zero
                } else if self.is_zero(rhs_src)
                    || rhs_node.as_int().map_or(false, |(_, value)| value == 0)
                {
                    // The right hand side is zero in both branches
                    zero_fact_true_branch(rhs_src, true);
                    zero_fact_false_branch(rhs_src, true);

                    // The left hand side is zero in the false branch and non-zero in the true branch
                    zero_fact_true_branch(lhs_src, false);
                    zero_fact_false_branch(lhs_src, true);
                }
            }

        // if x == y { ... } else { ... }
        } else if let Some(eq) = condition.as_eq() {
            let ((lhs_node, lhs_src, _), (rhs_node, rhs_src, _)) =
                (graph.get_input(eq.lhs()), graph.get_input(eq.rhs()));

            // If the left hand operand is zero
            if self.is_zero(lhs_src) || lhs_node.as_int().map_or(false, |(_, value)| value == 0) {
                // The left hand side is zero in both branches
                zero_fact_true_branch(lhs_src, true);
                zero_fact_false_branch(lhs_src, true);

                // The right hand side is zero in the true branch and non-zero in the false branch
                zero_fact_true_branch(rhs_src, true);
                zero_fact_false_branch(rhs_src, false);

            // If the right hand operand is zero
            } else if self.is_zero(rhs_src)
                || rhs_node.as_int().map_or(false, |(_, value)| value == 0)
            {
                // The right hand side is zero in both branches
                zero_fact_true_branch(rhs_src, true);
                zero_fact_false_branch(rhs_src, true);

                // The left hand side is zero in the true branch and non-zero in the false branch
                zero_fact_true_branch(lhs_src, true);
                zero_fact_false_branch(lhs_src, false);
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
        // FIXME: Some tape values can be preserved within the loop
        let mut visitor = Self::new(self.tape.len());

        for (input, param) in theta.invariant_input_pairs() {
            let source = graph.input_source(input);

            // Propagate facts
            if let Some(facts) = self.facts.get(&source).cloned() {
                visitor
                    .facts
                    .insert(param.output(), facts)
                    .debug_unwrap_none();
            }

            // Propagate constants
            if let Some(constant) = self.constants.get(&source).cloned() {
                visitor
                    .constants
                    .insert(param.output(), constant)
                    .debug_unwrap_none();
            }
        }

        changed |= visitor.visit_graph(theta.body_mut());

        // After a loop's body has completed, if the condition is `!(x == 0)` then `x` will
        // be zero afterwards
        if let Some(not) = theta
            .body()
            .input_source_node(theta.condition().input())
            .as_not()
        {
            if let Some(eq) = theta.body().input_source_node(not.input()).as_eq() {
                let (lhs_src, rhs_src) = (
                    theta.body().input_source(eq.lhs()),
                    theta.body().input_source(eq.rhs()),
                );

                let zeroed_val = if visitor.is_zero(lhs_src) {
                    Some(lhs_src)
                } else if visitor.is_zero(rhs_src) {
                    Some(rhs_src)
                } else {
                    None
                };

                if let Some(zeroed) = zeroed_val {
                    // If the zeroed value is an output param, propagate that directly
                    if let Some((output, _)) = theta
                        .output_pairs()
                        .find(|(_, param)| theta.body().input_source(param.input()) == rhs_src)
                    {
                        tracing::trace!(theta = ?theta.node(), "inferred that loop output variable {:?} was zero", output);

                        self.facts
                            .entry(output)
                            .and_modify(|fact| fact.is_zero = Some(true))
                            .or_insert_with(|| Facts {
                                is_zero: Some(true),
                                ..Default::default()
                            });
                    }

                    // If the value was loaded from an address, propagate that info to stores
                    if let Some(load) = theta
                        .body()
                        .cast_node::<Load>(theta.body().port_parent(zeroed))
                    {
                        let load_ptr_src = theta.body().input_source(load.ptr());
                        dbg!(load_ptr_src, load,);

                        if let Some(store) = theta
                            .body()
                            .input_source_node(theta.end_node().input_effect())
                            .as_store()
                        {
                            dbg!(
                                load_ptr_src,
                                load,
                                theta.body().input_source(store.ptr()),
                                store
                            );
                            if load_ptr_src == theta.body().input_source(store.ptr()) {
                                tracing::trace!(
                                    theta = ?theta.node(),
                                    "inferred that zeroed store to address {:?} was zero",
                                    load_ptr_src,
                                );
                            }
                        }
                    }
                }
            }
        }

        if changed {
            graph.replace_node(theta.node(), theta);
            self.changed();
        }
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: i32) {
        self.constants.insert(int.value(), value.into());

        if value == 0 {
            tracing::trace!("got zero value {:?}", int.value());

            self.facts
                .entry(int.value())
                .and_modify(|facts| facts.is_zero = Some(true))
                .or_insert_with(|| Facts {
                    is_zero: Some(true),
                    ..Default::default()
                });
        } else {
            self.facts
                .entry(int.value())
                .and_modify(|facts| facts.is_zero = Some(false))
                .or_insert_with(|| Facts {
                    is_zero: Some(false),
                    ..Default::default()
                });
        }

        if value.is_positive() {
            self.facts
                .entry(int.value())
                .and_modify(|facts| facts.sign = Some(Sign::Positive))
                .or_insert_with(|| Facts {
                    sign: Some(Sign::Positive),
                    ..Default::default()
                });
        } else if value.is_negative() {
            self.facts
                .entry(int.value())
                .and_modify(|facts| facts.sign = Some(Sign::Negative))
                .or_insert_with(|| Facts {
                    sign: Some(Sign::Negative),
                    ..Default::default()
                });
        }

        if value.rem_euclid(2) == 1 {
            self.facts
                .entry(int.value())
                .and_modify(|facts| facts.parity = Some(Parity::Even))
                .or_insert_with(|| Facts {
                    parity: Some(Parity::Even),
                    ..Default::default()
                });
        } else {
            self.facts
                .entry(int.value())
                .and_modify(|facts| facts.parity = Some(Parity::Odd))
                .or_insert_with(|| Facts {
                    parity: Some(Parity::Odd),
                    ..Default::default()
                });
        }
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        self.constants.insert(bool.value(), value.into());
    }

    fn visit_eq(&mut self, graph: &mut Rvsdg, eq: Eq) {
        if self.source_is_zero(graph, eq.lhs()) && self.source_is_zero(graph, eq.rhs()) {
            tracing::trace!(?eq, "replaced comparison with zero with true");

            let evaluated = graph.bool(true);
            graph.rewire_dependents(eq.value(), evaluated.value());

            self.changed();
        }
    }

    fn visit_add(&mut self, graph: &mut Rvsdg, add: Add) {
        let (lhs, rhs) = (graph.input_source(add.lhs()), graph.input_source(add.rhs()));

        if self.is_zero(lhs) {
            tracing::trace!(?add, "replaced addition by zero to the non-zero side");

            graph.rewire_dependents(add.value(), rhs);
            self.changed();
        } else if self.is_zero(rhs) {
            tracing::trace!(?add, "replaced addition by zero to the non-zero side");

            graph.rewire_dependents(add.value(), lhs);
            self.changed();
        }
    }

    fn visit_store(&mut self, graph: &mut Rvsdg, store: Store) {
        if let Some(ptr) = self
            .constants
            .get(&graph.input_source(store.ptr()))
            .and_then(Const::convert_to_i32)
        {
            let value_src = graph.input_source(store.value());
            let value_facts = self.facts.get(&value_src).cloned();
            let value_consts = self.constants.get(&value_src).cloned();

            let ptr = ptr.rem_euclid(self.tape.len() as i32) as usize;
            self.tape[ptr] = (value_facts, value_consts);
        } else {
            for cell in &mut self.tape {
                *cell = (None, None);
            }
        }
    }

    fn visit_load(&mut self, graph: &mut Rvsdg, load: Load) {
        if let Some(ptr) = self
            .constants
            .get(&graph.input_source(load.ptr()))
            .and_then(Const::convert_to_i32)
        {
            let ptr = ptr.rem_euclid(self.tape.len() as i32) as usize;
            let (facts, consts) = self.tape[ptr].clone();

            if let Some(facts) = facts {
                self.facts.insert(load.value(), facts);
            }
            if let Some(consts) = consts {
                self.constants.insert(load.value(), consts);
            }
        }
    }
}
