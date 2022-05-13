use crate::{
    graph::{Add, Bool, Byte, Eq, Gamma, Int, Mul, Neq, NodeExt, Not, OutputPort, Rvsdg, Theta},
    passes::{
        utils::{ChangeReport, Changes},
        Pass,
    },
    values::{Cell, Ptr},
};
use std::{collections::BTreeSet, mem::swap};
use tinyvec::TinyVec;

#[derive(Debug)]
pub struct Canonicalize {
    constants: BTreeSet<OutputPort>,
    changes: Changes<3>,
}

impl Canonicalize {
    pub fn new() -> Self {
        Self {
            constants: BTreeSet::new(),
            changes: Changes::new(["canonicalizations", "eq-to-neq", "neq-to-eq"]),
        }
    }
}

impl Pass for Canonicalize {
    fn pass_name(&self) -> &'static str {
        "canonicalize"
    }

    fn did_change(&self) -> bool {
        self.changes.did_change()
    }

    fn reset(&mut self) {
        self.constants.clear();
        self.changes.reset();
    }

    fn report(&self) -> ChangeReport {
        self.changes.as_report()
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, _: Ptr) {
        self.constants.insert(int.value());
    }

    fn visit_byte(&mut self, _graph: &mut Rvsdg, byte: Byte, _: Cell) {
        self.constants.insert(byte.value());
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, _: bool) {
        self.constants.insert(bool.value());
    }

    fn visit_add(&mut self, graph: &mut Rvsdg, add: Add) {
        let (lhs_src, rhs_src) = (graph.input_source(add.lhs()), graph.input_source(add.rhs()));

        if self.constants.contains(&lhs_src) && !self.constants.contains(&rhs_src) {
            tracing::debug!(
                ?add,
                ?lhs_src,
                ?rhs_src,
                "swapping add inputs {} and {}",
                add.lhs(),
                add.rhs(),
            );

            graph.remove_input_edges(add.lhs());
            graph.remove_input_edges(add.rhs());

            graph.add_value_edge(rhs_src, add.lhs());
            graph.add_value_edge(lhs_src, add.rhs());

            self.changes.inc::<"canonicalizations">();
        }
    }

    // Note: We can't canonicalize subtraction

    fn visit_mul(&mut self, graph: &mut Rvsdg, mul: Mul) {
        let (lhs_src, rhs_src) = (graph.input_source(mul.lhs()), graph.input_source(mul.rhs()));

        if self.constants.contains(&lhs_src) && !self.constants.contains(&rhs_src) {
            tracing::debug!(
                ?mul,
                ?lhs_src,
                ?rhs_src,
                "swapping mul inputs {} and {}",
                mul.lhs(),
                mul.rhs(),
            );

            graph.remove_input_edges(mul.lhs());
            graph.remove_input_edges(mul.rhs());

            graph.add_value_edge(rhs_src, mul.lhs());
            graph.add_value_edge(lhs_src, mul.rhs());

            self.changes.inc::<"canonicalizations">();
        }
    }

    fn visit_eq(&mut self, graph: &mut Rvsdg, eq: Eq) {
        let (mut lhs_src, mut rhs_src) =
            (graph.input_source(eq.lhs()), graph.input_source(eq.rhs()));

        // If the lhs is a constant and the rhs isn't, swap the two inputs
        // to turn `eq 10, x` into `eq x, 10`. This doesn't affect eqs
        // with two constant or two non-constant inputs
        if self.constants.contains(&lhs_src) && !self.constants.contains(&rhs_src) {
            tracing::debug!(
                ?eq,
                ?lhs_src,
                ?rhs_src,
                "swapping eq inputs {} and {}",
                eq.lhs(),
                eq.rhs(),
            );

            graph.remove_input_edges(eq.lhs());
            graph.remove_input_edges(eq.rhs());

            graph.add_value_edge(rhs_src, eq.lhs());
            graph.add_value_edge(lhs_src, eq.rhs());

            // Swap the two inputs for when we do eq => neq canonicalization
            swap(&mut lhs_src, &mut rhs_src);

            self.changes.inc::<"canonicalizations">();
        }

        // Find all consumers of this eq that are `not` nodes.
        // This will find `not (eq x, y)`
        let consuming_nots: TinyVec<[_; 2]> = graph
            .cast_output_consumers::<Not>(eq.value())
            .copied()
            .collect();

        // If any of the consumers are nots, create a neq node and rewire all consumers of the `not`
        // to the `neq`, transforming
        //
        // ```
        // x = eq a, b
        // y = not x
        // ```
        //
        // Into this
        //
        // ```
        // y = neq a, b
        // ```
        let mut neq = None;
        for not in consuming_nots {
            let neq = *neq.get_or_insert_with(|| graph.neq(lhs_src, rhs_src).value());
            graph.rewire_dependents(not.value(), neq);

            self.changes.inc::<"eq-to-neq">();
        }
    }

    fn visit_neq(&mut self, graph: &mut Rvsdg, neq: Neq) {
        let (mut lhs_src, mut rhs_src) =
            (graph.input_source(neq.lhs()), graph.input_source(neq.rhs()));

        if self.constants.contains(&lhs_src) && !self.constants.contains(&rhs_src) {
            tracing::debug!(
                ?neq,
                ?lhs_src,
                ?rhs_src,
                "swapping neq inputs {} and {}",
                neq.lhs(),
                neq.rhs(),
            );

            graph.remove_input_edges(neq.lhs());
            graph.remove_input_edges(neq.rhs());

            graph.add_value_edge(rhs_src, neq.lhs());
            graph.add_value_edge(lhs_src, neq.rhs());

            // Swap the two inputs for when we do neq => eq canonicalization
            swap(&mut lhs_src, &mut rhs_src);

            self.changes.inc::<"canonicalizations">();
        }

        // Find all consumers of this eq that are `not` nodes.
        // This will find `not (neq x, y)`
        let consuming_nots: TinyVec<[_; 2]> = graph
            .cast_output_consumers::<Not>(neq.value())
            .copied()
            .collect();

        // If any of the consumers are nots, create an eq node and rewire all consumers of the `not`
        // to the `eq`, transforming
        //
        // ```
        // x = neq a, b
        // y = not x
        // ```
        //
        // Into this
        //
        // ```
        // y = eq a, b
        // ```
        let mut eq = None;
        for not in consuming_nots {
            let eq = *eq.get_or_insert_with(|| graph.eq(lhs_src, rhs_src).value());
            graph.rewire_dependents(not.value(), eq);

            self.changes.inc::<"neq-to-eq">();
        }
    }

    // FIXME: Remove the added constants from the inner scopes
    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;
        changed |= self.visit_graph(gamma.true_mut());
        changed |= self.visit_graph(gamma.false_mut());

        if changed {
            graph.replace_node(gamma.node(), gamma);
        }
    }

    // FIXME: Remove the added constants from the inner scopes
    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        if self.visit_graph(theta.body_mut()) {
            graph.replace_node(theta.node(), theta);
        }
    }
}
