use crate::{
    graph::{EdgeKind, Gamma, NodeExt, Rvsdg, Store, Theta},
    passes::Pass,
};

/// Removes unobserved stores
pub struct UnobservedStore {
    changed: bool,
    // TODO: This is also a bit ham-fisted, but a more complex analysis of
    //       loop invariant cells would be required for anything better
    within_theta: bool,
    within_gamma: bool,
}

impl UnobservedStore {
    pub fn new() -> Self {
        Self {
            changed: false,
            within_theta: false,
            within_gamma: false,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }
}

impl Pass for UnobservedStore {
    fn pass_name(&self) -> &str {
        "unobserved-store"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.changed = false;
        self.within_theta = false;
    }

    fn visit_store(&mut self, graph: &mut Rvsdg, store: Store) {
        if let Some((consumer, _, kind)) = graph.get_output(store.effect()) {
            debug_assert_eq!(kind, EdgeKind::Effect);

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
            // So ideally we should try to follow the effect chain to try and remove
            // as many redundant stores as possible
            let stores_to_identical_cell = consumer.as_store().map_or(false, |consumer| {
                graph.get_input(consumer.ptr()).1 == graph.get_input(store.ptr()).1
            });

            let consumer_is_end = consumer.is_end() && !self.within_theta && !self.within_gamma;
            let consumer_is_infinite = consumer.as_theta().map_or(false, Theta::is_infinite);

            if consumer_is_end || stores_to_identical_cell || consumer_is_infinite {
                tracing::debug!(
                    consumer_is_end,
                    consumer_is_infinite,
                    stores_to_identical_cell,
                    "removing unobserved store {:?}",
                    store,
                );

                graph.splice_ports(store.effect_in(), store.effect());
                graph.remove_node(store.node());
                self.changed();
            }
        }
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;

        let (mut truthy_visitor, mut falsy_visitor) = (Self::new(), Self::new());
        truthy_visitor.within_gamma = true;
        falsy_visitor.within_gamma = true;
        truthy_visitor.within_theta = self.within_theta;
        falsy_visitor.within_theta = self.within_theta;

        changed |= truthy_visitor.visit_graph(gamma.true_mut());
        changed |= falsy_visitor.visit_graph(gamma.false_mut());

        if changed {
            graph.replace_node(gamma.node(), gamma);
            self.changed();
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut changed = false;

        let mut visitor = Self::new();
        visitor.within_theta = true;
        visitor.within_gamma = self.within_gamma;

        changed |= visitor.visit_graph(theta.body_mut());

        if changed {
            graph.replace_node(theta.node(), theta);
            self.changed();
        }
    }
}

impl Default for UnobservedStore {
    fn default() -> Self {
        Self::new()
    }
}
