use crate::{
    graph::{Bool, NodeExt, Rvsdg, Theta},
    passes::{
        dataflow::{domain::Domain, Dataflow},
        Pass,
    },
};

impl Dataflow {
    pub(super) fn compute_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut theta_changed = false;

        // TODO: Perform induction on loop variables where we can
        let inputs = self.collect_subgraph_inputs(graph, theta.body(), theta.input_pair_ids());

        // We can try to do induction on top-level thetas
        // (thetas without any thetas in their bodies)
        if !theta.has_child_thetas() {}

        // The visitor we create for the theta's body cannot mutate, as we need to
        // iterate until a fixpoint is reached and *then* optimize based off of that fixpoint state
        // TODO: Propagate constraints into the visitor
        let mut visitor = self.clone_for_subscope(inputs).with_mutation(false);
        // Visit the theta's body once before starting the fixpoint
        visitor.visit_graph(theta.body_mut());

        // Apply feedback parameters
        for (input, output) in theta.variant_inputs_loopback() {
            let output_source = theta.body().input_source(output.input());
            if let Some(output_domain) = visitor.domain(output_source).cloned() {
                visitor.add_domain(input.output(), output_domain);
            }
        }

        // Check if the theta can possibly iterate more than once
        let condition_src = theta.body().input_source(theta.condition().input());
        let can_iterate = visitor
            .domain(condition_src)
            .and_then(Domain::as_bool_set)
            // If it's possible for the theta's condition to be true, we must
            // iterate on it
            .map_or(true, |domain| domain.contains(true));

        // Note that we don't apply any constraints to the theta's initial inputs, this
        // is because the theta's condition is checked *after* each iteration

        // If the theta can iterate any, run its body to fixpoint
        if can_iterate {
            let (mut accrued_tape, mut accrued_values, mut fixpoint_iters) =
                (visitor.tape.clone(), visitor.values.clone(), 1usize);

            loop {
                // Visit the theta's body
                visitor.visit_graph(theta.body_mut());

                let mut did_change = false;

                // Union the accrued tape and this iteration's tape
                did_change |= accrued_tape.union(&visitor.tape);

                // Apply feedback parameters
                for (input, output) in theta.variant_inputs_loopback() {
                    let output_source = theta.body().input_source(output.input());
                    if let Some(output_domain) = visitor.domain(output_source).cloned() {
                        visitor.add_domain(input.output(), output_domain);
                    }
                }

                // Union all of the values
                accrued_values =
                    accrued_values.union_with(visitor.values.clone(), |mut accrued, mut domain| {
                        did_change |= accrued.union_mut(&mut domain);
                        accrued
                    });

                // Changes occurred, continue iterating
                if did_change {
                    fixpoint_iters += 1;

                // Otherwise we've hit a fixpoint!
                } else {
                    tracing::debug!(
                        "theta {} hit fixpoint in {fixpoint_iters} iteration{}",
                        theta.node(),
                        if fixpoint_iters == 1 { "" } else { "s" },
                    );
                    break;
                }
            }

        // If the theta's body can't ever run more than once,
        // change its condition to false to indicate that.
        // Additionally we want to make sure the theta's condition
        // isn't already a boolean literal so that we don't cause
        // more churn than we have to
        } else if self.can_mutate && theta.body().cast_parent::<_, Bool>(condition_src).is_none() {
            let cond = theta.body_mut().bool(false);
            theta
                .body_mut()
                .rewire_dependents(condition_src, cond.value());

            self.changes.inc::<"const-theta-cond">();
            theta_changed = true;
        }

        if self.can_mutate {
            // Finally, after running the body to a fixpoint we can optimize the innards with
            // the fixpoint-ed values
            visitor.allow_mutation();
            theta_changed |= visitor.visit_graph(theta.body_mut());
            self.changes.combine(&visitor.changes);
        }

        // TODO: Pull out fixpoint-ed constraints from the body

        // Pull output parameters into the outer scope
        for (output, param) in theta.output_pairs() {
            let param_source = theta.body().input_source(param.input());
            if let Some(domain) = visitor.domain(param_source).cloned() {
                self.add_domain(output, domain);
            }
        }

        // Use the body's fixpoint-ed tape as our tape
        self.tape = visitor.tape;

        // If we've changed anything, replace our node within the graph
        if theta_changed && self.can_mutate {
            graph.replace_node(theta.node(), theta);
        }
    }
}
