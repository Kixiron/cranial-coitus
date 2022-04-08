use crate::{
    graph::{Bool, Gamma, InputParam, NodeExt, OutputParam, Rvsdg},
    passes::{
        dataflow::{domain::Domain, Dataflow},
        Pass,
    },
};
use std::mem::swap;

impl Dataflow {
    pub(super) fn compute_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let cond = graph.input_source(gamma.condition());
        let (true_reachable, false_reachable) = self
            .domain(cond)
            .and_then(Domain::as_bool_set)
            .map(|domain| (domain.contains(true), domain.contains(false)))
            .unwrap_or((true, true));

        // Log any unreachable branches and record a change
        if self.can_mutate && (!true_reachable || !false_reachable) {
            let cond_src_not_bool = graph.cast_parent::<_, Bool>(cond).is_none();

            // If the false branch isn't reachable change the condition to false
            if !true_reachable && cond_src_not_bool {
                tracing::debug!("elided true branch for gamma {}", gamma.node());

                let bool = graph.bool(false);
                graph.rewire_dependents(cond, bool.value());

                self.changes.inc::<"gamma-branch-elision">();

            // If the false branch isn't reachable change the condition to true
            } else if !false_reachable && cond_src_not_bool {
                tracing::debug!("elided false branch for gamma {}", gamma.node());

                let bool = graph.bool(true);
                graph.rewire_dependents(cond, bool.value());

                self.changes.inc::<"gamma-branch-elision">();
            }

            // If neither are reachable then we're in a very weird situation
            if !true_reachable && !false_reachable {
                tracing::debug!("neither branch of gamma {} is reachable", gamma.node());
            }
        }

        // Create the visitor and input values for each branch, skipping unreachable branches
        // FIXME: Propagate constraints
        let mut true_visitor = true_reachable.then(|| {
            let values =
                self.collect_subgraph_inputs(graph, gamma.true_branch(), gamma.true_input_pairs());
            self.clone_for_subscope(values)
                .with_mutation(self.can_mutate)
        });
        let mut false_visitor = false_reachable.then(|| {
            let values = self.collect_subgraph_inputs(
                graph,
                gamma.false_branch(),
                gamma.false_input_pairs(),
            );
            self.clone_for_subscope(values)
                .with_mutation(self.can_mutate)
        });

        // Apply any constraints from the gamma's condition
        // Note that we don't use `.add_domain()` since it *unions* and we want
        // to overwrite the value, our constrained values are more narrow than
        // the domains they currently hold
        if let Some(visitor) = true_visitor.as_mut() {
            for (input, param) in gamma.true_input_pairs() {
                let input_source = graph.input_source(input);

                if let Some((true_domain, _)) = self.constraints.get(&(cond, input_source)) {
                    let param = gamma.true_branch().cast_node::<InputParam>(param).unwrap();

                    let mut true_domain = true_domain.clone();
                    visitor
                        .values
                        .entry(param.output())
                        .and_modify(|domain| {
                            domain.union_mut(&mut true_domain);
                        })
                        .or_insert(true_domain);
                }
            }
        }
        if let Some(visitor) = false_visitor.as_mut() {
            for (input, param) in gamma.false_input_pairs() {
                let input_source = graph.input_source(input);

                if let Some((_, false_domain)) = self.constraints.get(&(cond, input_source)) {
                    let param = gamma.false_branch().cast_node::<InputParam>(param).unwrap();

                    let mut false_domain = false_domain.clone();
                    visitor
                        .values
                        .entry(param.output())
                        .and_modify(|domain| {
                            domain.union_mut(&mut false_domain);
                        })
                        .or_insert(false_domain);
                }
            }
        }

        // Visit all reachable branches of the gamma node
        let mut changed = false;
        if let Some(visitor) = true_visitor.as_mut() {
            changed |= visitor.visit_graph(gamma.true_mut());
            self.changes.combine(&visitor.changes);
        }
        if let Some(visitor) = false_visitor.as_mut() {
            changed |= visitor.visit_graph(gamma.false_mut());
            self.changes.combine(&visitor.changes);
        }

        // Unify the values of both tapes
        // Note: Don't use `true_visitor` or `false_visitor`'s tape after this, they're
        //       both invalidated
        match (
            true_visitor.as_mut().map(|visitor| &mut visitor.tape),
            false_visitor.as_mut().map(|visitor| &mut visitor.tape),
        ) {
            // If both branches are reachable, union each cell value across branches
            (Some(true_tape), Some(false_tape)) => {
                true_tape.union_into(false_tape, &mut self.tape);
            }

            // If only one branch is reachable, the output tape will the same as that branch's
            (Some(tape), None) | (None, Some(tape)) => swap(&mut self.tape, tape),

            // TODO: What do we do here when both branches are unreachable?
            (None, None) => {}
        }

        // TODO: We can't use mutable references here since multiple output
        //       params could have the same input node, should probably check
        //       on that when possible
        // TODO: Propagate & union constraints
        match (
            true_visitor.as_ref().map(|visitor| (&visitor.values, true)),
            false_visitor
                .as_ref()
                .map(|visitor| (&visitor.values, false)),
        ) {
            (Some((values, is_true_branch)), None) | (None, Some((values, is_true_branch))) => {
                let branch_graph = if is_true_branch {
                    gamma.true_branch()
                } else {
                    gamma.false_branch()
                };
                let output_domain = |param| {
                    let param = branch_graph.cast_node::<OutputParam>(param).unwrap();
                    let source = branch_graph.input_source(param.input());
                    values.get(&source)
                };

                if is_true_branch {
                    for (output, param) in gamma.true_output_pairs() {
                        if let Some(output_domain) = output_domain(param) {
                            self.add_domain(output, output_domain.clone());
                        }
                    }
                } else {
                    for (output, param) in gamma.false_output_pairs() {
                        if let Some(output_domain) = output_domain(param) {
                            self.add_domain(output, output_domain.clone());
                        }
                    }
                }
            }

            // Unify the possible values of each output
            (Some((true_values, _)), Some((false_values, _))) => {
                for (output, [true_param, false_param]) in gamma.paired_outputs() {
                    let true_param = gamma
                        .true_branch()
                        .cast_node::<OutputParam>(true_param)
                        .unwrap();
                    let true_src = gamma.true_branch().input_source(true_param.input());
                    let true_domain = true_values.get(&true_src);

                    let false_param = gamma
                        .false_branch()
                        .cast_node::<OutputParam>(false_param)
                        .unwrap();
                    let false_src = gamma.false_branch().input_source(false_param.input());
                    let false_domain = false_values.get(&false_src);

                    if let (Some(true_domain), Some(false_domain)) = (true_domain, false_domain) {
                        let mut domain = true_domain.clone();
                        domain.union(false_domain);
                        self.add_domain(output, domain);
                    }
                }
            }

            // TODO: If both branches are unreachable we have no output info
            (None, None) => {}
        }

        // If we've changed anything, replace the gamma node
        if changed && self.can_mutate {
            // TODO: Branch inlining
            match (true_reachable, false_reachable) {
                (true, true) => {} // graph.replace_node(gamma.node(), gamma);

                // TODO: Inline the true branch
                (true, false) => {}

                // TODO: Inline the false branch
                (false, true) => {}

                // TODO: Replace with an unreachable node?
                (false, false) => {}
            }

            graph.replace_node(gamma.node(), gamma);
        }
    }
}
