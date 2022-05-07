use crate::{
    graph::{Bool, Byte, EdgeKind, Gamma, Int, NodeExt, Rvsdg, Store, Theta},
    passes::{
        utils::{ChangeReport, Changes, ConstantStore},
        Pass,
    },
    values::{Cell, Ptr},
};

/// Removes unobserved stores
// FIXME: Every time an unobserved store is removed, a new node becomes the last effect.
//        This isn't bad, but it requires the pass to run many many times before
//        all of the unobserved effects have been removed. Instead, we want to do
//        batched removal where we remove *all* of the trailing stores instead of
//        removing them one at a time.
pub struct UnobservedStore {
    // TODO: This is also a bit ham-fisted, but a more complex analysis of
    //       loop invariant cells would be required for anything better
    within_theta: bool,
    within_gamma: bool,
    constants: ConstantStore,
    changes: Changes<1>,
}

impl UnobservedStore {
    pub fn new(tape_len: u16) -> Self {
        Self {
            within_theta: false,
            within_gamma: false,
            constants: ConstantStore::new(tape_len),
            changes: Changes::new(["elided-stores"]),
        }
    }
}

impl Pass for UnobservedStore {
    fn pass_name(&self) -> &str {
        "unobserved-store"
    }

    fn did_change(&self) -> bool {
        self.changes.did_change()
    }

    fn reset(&mut self) {
        self.changes.reset();
        self.constants.clear();
        self.within_theta = false;
        self.within_gamma = false;
    }

    fn report(&self) -> ChangeReport {
        self.changes.as_report()
    }

    // If the effect consumer can't observe the store, remove this store.
    // This happens when the next effect is either an end node or a store to
    // the same address.
    //
    // This addresses these two cases:
    //
    // ```
    // block {
    //   store _0, _1 // Removes this store, nothing's after it to observe it
    // }
    // ```
    //
    // Successive stores to the same address are redundant, only the last one
    // can be observed
    //
    // ```
    // store _0, _1
    // store _0, _1
    // store _0, _1
    // ```
    //
    // TODO: This can get broken up by a sequence like this:
    // ```
    // store _0, _2
    // store _1, _2
    // store _0, _2
    // ```
    fn visit_store(&mut self, graph: &mut Rvsdg, store: Store) {
        let store_ptr = graph.input_source(store.ptr());
        let store_ptr_value = self.constants.ptr(store_ptr);

        let mut last_effect = store.output_effect();
        while let Some((consumer, _, kind)) = graph.get_output(last_effect) {
            debug_assert_eq!(kind, EdgeKind::Effect);

            // If the two stores go to the same cell, we can remove the first store
            let stores_to_identical_cell = consumer
                .as_store()
                .map(|consumer| graph.input_source(consumer.ptr()) == store_ptr)
                .unwrap_or(false);

            // TODO: We'd like to have more robust stuff for this to allow getting rid of redundant
            //       stores within gammas, for instance
            let consumer_is_end = consumer.is_end() && !self.within_theta && !self.within_gamma;

            // If the consumer is a load from a totally different cell, we can keep scanning forward
            let loads_from_different_cell = consumer
                .as_load()
                .and_then(|load| {
                    Some((
                        load.output_effect(),
                        self.constants.ptr(graph.input_source(load.ptr()))?,
                    ))
                })
                .zip(store_ptr_value)
                .and_then(|((load_effect, load_ptr_value), store_ptr_value)| {
                    (load_ptr_value != store_ptr_value).then_some(load_effect)
                });

            // If the consumer is a store to a totally different cell, we can keep scanning forward
            let stores_to_different_cell = consumer
                .as_store()
                .and_then(|consumer| {
                    Some((
                        consumer.output_effect(),
                        self.constants.ptr(graph.input_source(consumer.ptr()))?,
                    ))
                })
                .zip(store_ptr_value)
                .and_then(|((consumer_effect, consumer_ptr_value), store_ptr_value)| {
                    (consumer_ptr_value != store_ptr_value).then_some(consumer_effect)
                });

            // If the consumer is an I/O operation then it doesn't affect memory
            let is_io_operation = consumer
                .as_output()
                .map(|output| output.output_effect())
                .or_else(|| consumer.as_input().map(|input| input.output_effect()));

            if consumer_is_end || stores_to_identical_cell {
                tracing::debug!(
                    consumer_is_end,
                    stores_to_identical_cell,
                    "removing unobserved store {:?}",
                    store,
                );

                graph.splice_ports(store.effect_in(), store.output_effect());
                graph.remove_node(store.node());
                self.changes.inc::<"elided-stores">();

                break;

            // If the consumer doesn't affect program memory in a way relevant to the target store, we
            // can keep scanning past it
            } else if let Some(effect) = is_io_operation
                .or(loads_from_different_cell)
                .or(stores_to_different_cell)
            {
                tracing::debug!(
                    ?consumer,
                    ?is_io_operation,
                    ?loads_from_different_cell,
                    ?stores_to_different_cell,
                    "consumer {:?} of store {:?} doesn't affect memory in a way relevant to the store",
                    store.node(),
                    consumer.node(),
                );
                last_effect = effect;

            // Otherwise this is an operation that affects program memory, so we can't remove stores past it
            } else {
                break;
            }
        }
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;

        let (mut true_visitor, mut false_visitor) = (
            Self::new(self.constants.tape_len()),
            Self::new(self.constants.tape_len()),
        );
        true_visitor.within_gamma = true;
        false_visitor.within_gamma = true;
        true_visitor.within_theta = self.within_theta;
        false_visitor.within_theta = self.within_theta;

        // Get constant inputs from the current context and propagate them into the branches
        self.constants.gamma_inputs_into(
            &gamma,
            graph,
            &mut true_visitor.constants,
            &mut false_visitor.constants,
        );

        changed |= true_visitor.visit_graph(gamma.true_mut());
        self.changes.combine(&true_visitor.changes);

        changed |= false_visitor.visit_graph(gamma.false_mut());
        self.changes.combine(&true_visitor.changes);

        if changed {
            graph.replace_node(gamma.node(), gamma);
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut changed = false;

        let mut visitor = Self::new(self.constants.tape_len());
        visitor.within_theta = true;
        visitor.within_gamma = self.within_gamma;

        // Get invariant inputs from the current context and place them into the
        // theta body's context
        self.constants
            .theta_invariant_inputs_into(&theta, graph, &mut visitor.constants);

        changed |= visitor.visit_graph(theta.body_mut());
        self.changes.combine(&visitor.changes);

        if changed {
            graph.replace_node(theta.node(), theta);
        }
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: Ptr) {
        self.constants.add(int.value(), value);
    }

    fn visit_byte(&mut self, _graph: &mut Rvsdg, byte: Byte, value: Cell) {
        self.constants.add(byte.value(), value);
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        self.constants.add(bool.value(), value);
    }
}
