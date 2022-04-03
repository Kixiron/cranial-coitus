use crate::{
    graph::{Eq, Neq, Rvsdg},
    passes::dataflow::{
        domain::{differential_product, BoolSet},
        Dataflow,
    },
};

impl Dataflow {
    pub(super) fn compute_eq(&mut self, graph: &mut Rvsdg, eq: Eq) {
        let (lhs_src, rhs_src) = (graph.input_source(eq.lhs()), graph.input_source(eq.rhs()));

        let mut eq_domain = BoolSet::full();
        let constraints =
            if let (Some(lhs), Some(rhs)) = (self.domain(lhs_src), self.domain(rhs_src)) {
                let intersection = lhs.intersect(rhs);

                // TODO: Skip calculating this for branches that are unreachable
                let (lhs_false_domain, rhs_false_domain) = differential_product(lhs, rhs);

                // If there's zero overlap between the two values, the comparison will always be false
                if intersection.is_empty() {
                    eq_domain.remove(true);

                // If both sides of the comparison are the same, then it'll always be true
                } else if lhs
                    .as_singleton()
                    .zip(rhs.as_singleton())
                    .map_or(false, |(lhs, rhs)| lhs == rhs)
                {
                    eq_domain.remove(false);
                }

                Some((intersection, lhs_false_domain, rhs_false_domain))
            } else {
                None
            };

        // Apply constraints if we got any
        if let Some((intersection, lhs_false_domain, rhs_false_domain)) = constraints {
            self.add_constraints(eq.value(), lhs_src, intersection.clone(), lhs_false_domain);
            self.add_constraints(eq.value(), rhs_src, intersection, rhs_false_domain);
        }

        // If the eq's result is statically known, replace the eq node
        // with a constant value
        if self.can_mutate {
            if let Some(result) = eq_domain.as_singleton() {
                let node = graph.bool(result);
                self.add_domain(node.value(), result);
                graph.rewire_dependents(eq.value(), node.value());
                self.changes.inc::<"const-eq">();
            }
        }

        self.add_domain(eq.value(), eq_domain);
    }

    pub(super) fn compute_neq(&mut self, graph: &mut Rvsdg, neq: Neq) {
        let (lhs_src, rhs_src) = (graph.input_source(neq.lhs()), graph.input_source(neq.rhs()));

        let mut neq_domain = BoolSet::full();
        let constraints =
            if let (Some(lhs), Some(rhs)) = (self.domain(lhs_src), self.domain(rhs_src)) {
                let intersection = lhs.intersect(rhs);

                // TODO: Skip calculating this for branches that are unreachable
                let (lhs_true_domain, rhs_true_domain) = differential_product(lhs, rhs);

                // If there's zero overlap between the two values, the comparison will always be true
                if intersection.is_empty() {
                    neq_domain.remove(false);

                // If both sides of the comparison are the same, then it'll always be false
                } else if lhs
                    .as_singleton()
                    .zip(rhs.as_singleton())
                    .map_or(false, |(lhs, rhs)| lhs == rhs)
                {
                    neq_domain.remove(true);
                }

                Some((lhs_true_domain, rhs_true_domain, intersection))
            } else {
                None
            };

        // Apply constraints if we got any
        if let Some((lhs_true_domain, rhs_true_domain, intersection)) = constraints {
            self.add_constraints(neq.value(), lhs_src, lhs_true_domain, intersection.clone());
            self.add_constraints(neq.value(), rhs_src, rhs_true_domain, intersection);
        }

        // If the neq's result is statically known, replace the neq node
        // with a constant value
        if self.can_mutate {
            if let Some(result) = neq_domain.as_singleton() {
                let node = graph.bool(result);
                self.add_domain(node.value(), result);
                graph.rewire_dependents(neq.value(), node.value());
                self.changes.inc::<"const-neq">();
            }
        }

        self.add_domain(neq.value(), neq_domain);
    }
}
