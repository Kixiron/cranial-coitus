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
        if let (Some(lhs), Some(rhs)) = (self.domain(lhs_src), self.domain(rhs_src)) {
            if let Some((lhs, rhs)) = lhs.as_singleton().zip(rhs.as_singleton()) {
                // lhs ≡ rhs ⟹ ¬(lhs ≠ rhs)
                if lhs == rhs {
                    // Only the true branch is reachable
                    eq_domain.remove(false);

                // lhs ≠ rhs ⟹ ¬(lhs ≡ rhs)
                } else {
                    // Only the false branch is reachable
                    eq_domain.remove(true);
                }

            // If it's possible that the two operands can have the same value,
            // then it's possible for the eq to return true
            } else if lhs.intersects(rhs) {
                // Both branches are reachable right now

                let intersection = lhs.intersect(rhs);
                let (lhs_true_domain, rhs_true_domain) = differential_product(lhs, rhs);
                debug_assert!(!intersection.is_empty());

                // Apply constraints to the operands within branches on this condition
                // TODO: Warn if we get empty sets here
                if !lhs_true_domain.is_empty() {
                    self.add_constraints(
                        eq.value(),
                        lhs_src,
                        intersection.clone(),
                        lhs_true_domain,
                    );
                }
                if !rhs_true_domain.is_empty() {
                    self.add_constraints(eq.value(), rhs_src, intersection, rhs_true_domain);
                }

            // If there's zero overlap between the two operands (lhs ∩ rhs = Ø)
            // then the eq can't possibly return true
            } else {
                // Only the false branch is reachable
                eq_domain.remove(true);
            }
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
        if let (Some(lhs), Some(rhs)) = (self.domain(lhs_src), self.domain(rhs_src)) {
            if let Some((lhs, rhs)) = lhs.as_singleton().zip(rhs.as_singleton()) {
                // lhs ≡ rhs ⟹ ¬(lhs ≠ rhs)
                if lhs == rhs {
                    // Only the false branch is reachable
                    neq_domain.remove(true);

                // lhs ≠ rhs ⟹ ¬(lhs ≡ rhs)
                } else {
                    // Only the true branch is reachable
                    neq_domain.remove(false);
                }

            // If it's possible that the two operands can have the same value,
            // then it's possible for the neq to return false
            } else if lhs.intersects(rhs) {
                // Both branches are reachable right now

                let intersection = lhs.intersect(rhs);
                let (lhs_false_domain, rhs_false_domain) = differential_product(lhs, rhs);
                debug_assert!(!intersection.is_empty());

                // Apply constraints to the operands within branches on this condition
                // TODO: Warn if we get empty sets here
                if !lhs_false_domain.is_empty() {
                    self.add_constraints(
                        neq.value(),
                        lhs_src,
                        lhs_false_domain,
                        intersection.clone(),
                    );
                }
                if !rhs_false_domain.is_empty() {
                    self.add_constraints(neq.value(), rhs_src, rhs_false_domain, intersection);
                }

            // If there's zero overlap between the two operands (lhs ∩ rhs = Ø)
            // then the neq can't possibly return false
            } else {
                // Only the true branch is reachable
                neq_domain.remove(false);
            }
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
