use crate::{
    graph::{NodeExt, Rvsdg, Theta},
    passes::{
        dataflow::{
            domain::{Domain, NormalizedDomains},
            Dataflow,
        },
        Pass,
    },
};

impl Dataflow {
    pub(super) fn compute_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        // let span = tracing::info_span!("theta fixpoint", theta = %theta.node());
        // let _span = span.enter();
        // tracing::info!("started fixpoint of theta {}", theta.node());

        let inputs = self.collect_subgraph_inputs(graph, theta.body(), theta.input_pair_ids());
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

        // TODO: We can actually check to see if the theta's condition is impossible to satisfy
        //       and if it is we can skip all of the fixpoint-ing

        // Note that we don't apply any constraints to the theta's initial inputs, this
        // is because the theta's condition is checked *after* each iteration

        {
            let (mut accrued_tape, mut accrued_values, mut fixpoint_iters) =
                (visitor.tape.clone(), visitor.values.clone(), 1usize);

            loop {
                // Visit the theta's body
                visitor.visit_graph(theta.body_mut());

                let mut did_change = false;

                // Iterate over every cell in the program tape, seeing if anything has changed
                assert_eq!(accrued_tape.len(), visitor.tape.len());
                for (accrued, cell) in accrued_tape.iter_mut().zip(visitor.tape.iter()) {
                    if accrued != cell {
                        let old = *accrued;
                        accrued.union_mut(&mut cell.clone());

                        // Make sure union-ing the values actually changed it,
                        // this won't happen in cases like `{ 1..10 } ∪ { 2..4 } = { 1..10 }`
                        if accrued != &old {
                            did_change = true;
                        }
                    }
                }

                // Apply feedback parameters
                for (input, output) in theta.variant_inputs_loopback() {
                    let output_source = theta.body().input_source(output.input());
                    if let Some(output_domain) = visitor.domain(output_source).cloned() {
                        visitor.add_domain(input.output(), output_domain);
                    }
                }

                // Union all of the values
                accrued_values =
                    accrued_values.union_with(visitor.values.clone(), |accrued, domain| {
                        match accrued.normalize(domain) {
                            NormalizedDomains::Bool(accrued, domain) => {
                                let mut new = accrued;
                                new.union(&domain);

                                if new != accrued {
                                    did_change = true;
                                }

                                Domain::Bool(new)
                            }

                            NormalizedDomains::Byte(mut accrued, mut domain) => {
                                if accrued != domain {
                                    let old = accrued;
                                    accrued.union_mut(&mut domain);

                                    if accrued != old {
                                        did_change = true;
                                    }
                                }

                                Domain::Byte(accrued)
                            }

                            NormalizedDomains::Int(mut accrued, mut domain) => {
                                if accrued != domain {
                                    let old = accrued.clone();
                                    accrued.union_mut(&mut domain);

                                    if accrued != old {
                                        did_change = true;
                                    }
                                }

                                Domain::Int(accrued)
                            }
                        }
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
        }

        // Finally, after running the body to a fixpoint we can optimize the innards with
        // the fixpoint-ed values
        visitor.allow_mutation();
        visitor.visit_graph(theta.body_mut());

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

        // tracing::info!("finished fixpoint of theta {}", theta.node());
    }
}
