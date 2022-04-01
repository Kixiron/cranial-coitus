use crate::{
    graph::{Load, Rvsdg, Store},
    passes::dataflow_v2::{domain::ByteSet, Dataflow},
};

impl Dataflow {
    // Note: This is an *incredibly* hot function so any optimizations we can possibly do
    //       go a really long way
    pub(super) fn compute_store(&mut self, graph: &mut Rvsdg, store: Store) {
        let (ptr, value) = (
            graph.input_source(store.ptr()),
            graph.input_source(store.value()),
        );
        let (ptr_domain, value_domain) = (self.values.get(&ptr), self.values.get(&value));

        match (ptr_domain, value_domain) {
            // If we don't know the pointer we're writing to or the value we're writing,
            // make every cell within the tape unknown
            (None, None) => self.tape.iter_mut().for_each(|cell| cell.make_full()),

            // If we don't know the pointer but we do know the value we're writing to it,
            // union every cell's value with the written value
            (None, Some(value)) => {
                let value = value.into_byte_set();
                self.tape.iter_mut().for_each(|cell| cell.union(value))
            }

            // If we know the pointer we're writing to but don't know what value
            // we're writing, we can simply make those cell values unknown
            (Some(ptr), None) => {
                let ptr = ptr.into_int_set(self.settings.tape_len);
                for ptr in ptr.iter() {
                    self.tape
                        .get_mut(ptr.value() as usize)
                        .expect("pointer values should be inbounds")
                        .make_full();
                }
            }

            // Otherwise if we know both the pointer and the written value,
            // we can replace the cell at each possible cell with our written value
            (Some(ptr), Some(value)) => {
                let (ptr, value) = (
                    ptr.into_int_set(self.settings.tape_len),
                    value.into_byte_set(),
                );

                for ptr in ptr.iter() {
                    let cell = self
                        .tape
                        .get_mut(ptr.value() as usize)
                        .expect("pointer values should be inbounds");
                    cell.clear();

                    *cell = value;
                }
            }
        }
    }

    pub(super) fn compute_load(&mut self, graph: &mut Rvsdg, load: Load) {
        let ptr = graph.input_source(load.ptr());
        let ptr_domain = self.values.get(&ptr);

        let loaded = match ptr_domain {
            Some(ptr) => {
                let ptr = ptr.into_int_set(self.settings.tape_len);
                if ptr.is_empty() {
                    tracing::error!(?load, %ptr, "loaded from pointer without any possible values?");
                    ByteSet::full()
                } else {
                    let mut ptr_values = ptr.iter();
                    let mut loaded = *self
                        .tape
                        .get(ptr_values.next().unwrap().value() as usize)
                        .expect("pointers should be inbounds");

                    // Make the loaded value the union of all cells we
                    // could possibly load
                    for ptr in ptr_values {
                        let cell = *self
                            .tape
                            .get(ptr.value() as usize)
                            .expect("pointers should be inbounds");
                        loaded.union(cell);
                    }

                    loaded
                }
            }

            // If we don't know the pointer we're loading from, return an unknown value
            // TODO: We could technically union the value of every cell within the tape,
            //       but is that actually going to give us any material gain?
            None => ByteSet::full(),
        };

        // If there's only one value we could have possibly loaded, replace this
        // load node with that value
        if self.can_mutate {
            if let Some(loaded) = loaded.as_singleton() {
                let node = graph.byte(loaded);
                self.add_domain(node.value(), loaded);
                graph.rewire_dependents(load.output_value(), node.value());
                self.changes.inc::<"const-load">();
            }
        }

        self.add_domain(load.output_value(), loaded);
    }
}
