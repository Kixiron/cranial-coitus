use crate::{
    graph::{Gamma, InputParam, NodeExt, OutputPort, Rvsdg, Theta},
    passes::Pass,
    utils::{AssertNone, HashMap},
};

/// Removes dead code from the graph
pub struct Dataflow {
    changed: bool,
    facts: HashMap<OutputPort, Facts>,
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
    pub fn new() -> Self {
        Self {
            changed: false,
            facts: HashMap::with_hasher(Default::default()),
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
        let (mut true_visitor, mut false_visitor) = (Self::new(), Self::new());

        for (&input, &[true_param, false_param]) in gamma.inputs().iter().zip(gamma.input_params())
        {
            if let Some(facts) = self.facts.get(&graph.input_source(input)).cloned() {
                let (true_param, false_param) = (
                    graph.to_node::<InputParam>(true_param),
                    graph.to_node::<InputParam>(false_param),
                );

                true_visitor
                    .facts
                    .insert(true_param.output(), facts.clone())
                    .debug_unwrap_none();
                false_visitor
                    .facts
                    .insert(false_param.output(), facts)
                    .debug_unwrap_none();
            }
        }

        let mut zero_fact_true_branch = |source, is_zero| {
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
        let mut visitor = Self::new();

        for (input, param) in theta.invariant_input_pairs() {
            if let Some(facts) = self.facts.get(&graph.input_source(input)).cloned() {
                visitor
                    .facts
                    .insert(param.output(), facts)
                    .debug_unwrap_none();
            }
        }

        changed |= visitor.visit_graph(theta.body_mut());

        if changed {
            graph.replace_node(theta.node(), theta);
            self.changed();
        }
    }
}

impl Default for Dataflow {
    fn default() -> Self {
        Self::new()
    }
}
