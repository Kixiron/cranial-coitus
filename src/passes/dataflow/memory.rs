use crate::{
    graph::{Load, Rvsdg, Scan, Store},
    passes::dataflow::{
        domain::{ByteSet, Domain},
        Dataflow,
    },
    values::Ptr,
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
            (None, None) => self.tape.fill(),

            // If we don't know the pointer but we do know the value we're writing to it,
            // union every cell's value with the written value
            (None, Some(value)) => {
                let value = value.into_byte_set();
                self.tape.iter_mut().for_each(|cell| {
                    cell.union(value);
                });
            }

            // If we know the pointer we're writing to but don't know what value
            // we're writing, we can simply make those cell values unknown
            (Some(ptr), None) => match ptr {
                Domain::Byte(bytes) => bytes.iter().for_each(|byte| {
                    let ptr = Ptr::new(byte as u16, self.tape_len());
                    self.tape[ptr].fill();
                }),
                Domain::Int(ints) => ints.iter().for_each(|ptr| self.tape[ptr].fill()),
                Domain::Bool(_) => unreachable!(),
            },

            // Otherwise if we know both the pointer and the written value,
            // we can replace the cell at each possible cell with our written value
            (Some(ptr), Some(value)) => {
                let value = value.into_byte_set();

                match ptr {
                    Domain::Byte(bytes) => bytes.iter().for_each(|byte| {
                        let ptr = Ptr::new(byte as u16, self.tape_len());
                        self.tape[ptr] = value;
                    }),
                    Domain::Int(ints) => ints.iter().for_each(|ptr| self.tape[ptr] = value),
                    Domain::Bool(_) => unreachable!(),
                }
            }
        }

        // TODO: Finer-grained provenance invalidation
        if self.can_mutate {
            self.port_provenance.clear();
        }
    }

    pub(super) fn compute_load(&mut self, graph: &mut Rvsdg, load: Load) {
        let ptr_source = graph.input_source(load.ptr());
        let ptr_domain = self.values.get(&ptr_source);

        let mut loaded = match ptr_domain {
            Some(ptr) => {
                let loaded = match ptr {
                    Domain::Byte(ptr) => {
                        if ptr.is_empty() {
                            tracing::error!(?load, %ptr, "loaded from pointer without any possible values?");
                            ByteSet::full()
                        } else {
                            // Make the loaded value the union of all cells we
                            // could possibly load
                            let mut loaded = ByteSet::empty();
                            for ptr in ptr.iter() {
                                let ptr = Ptr::new(ptr as u16, self.tape_len());
                                loaded.union(self.tape[ptr]);

                                if loaded.is_full() {
                                    break;
                                }
                            }

                            tracing::debug!(
                                "union across pointer range {ptr_source} produced loaded value {loaded}",
                            );
                            loaded
                        }
                    }

                    Domain::Int(ptr) => {
                        if ptr.is_empty() {
                            tracing::error!(?load, %ptr, "loaded from pointer without any possible values?");
                            ByteSet::full()
                        } else {
                            // Make the loaded value the union of all cells we
                            // could possibly load
                            let mut loaded = ByteSet::empty();
                            for ptr in ptr {
                                loaded.union(self.tape[ptr]);

                                if loaded.is_full() {
                                    break;
                                }
                            }

                            tracing::debug!(
                                "union across pointer range {ptr_source} produced loaded value {loaded}",
                            );
                            loaded
                        }
                    }

                    Domain::Bool(_) => unreachable!(),
                };

                if loaded.is_empty() {
                    ByteSet::full()
                } else {
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
            // If there's provenance for the given pointer, the loaded value
            // will be the intersection of both the provenance and the value
            // of the pointed to cells
            if let Some(provenance) = self.provenance(ptr_source) {
                tracing::debug!("intersecting provenance of {provenance} with loaded value {loaded} for pointer {ptr_source}");
                loaded = loaded.intersection(provenance);
            }

            if let Some(loaded) = loaded.as_singleton() {
                tracing::debug!(
                    "replaced load {} of pointer {ptr_source} with constant {loaded}",
                    load.output_value(),
                );

                let node = graph.byte(loaded);
                self.add_domain(node.value(), loaded);

                graph.rewire_dependents(load.output_value(), node.value());
                graph.splice_ports(load.input_effect(), load.output_effect());

                self.changes.inc::<"const-load">();
            }
        }

        self.add_domain(load.output_value(), loaded);
    }

    pub(super) fn compute_scan(&mut self, graph: &mut Rvsdg, scan: Scan) {
        // If we can find the domain of the needle value, add that provenance
        // to the output pointer
        if self.can_mutate
            && let Some(needle) = self.domain(graph.input_source(scan.needle())).map(Domain::into_byte_set)
            && !needle.is_full()
            && !needle.is_empty()
        {
            tracing::debug!("added scan provenance to output pointer {}: {needle}", scan.output_ptr());
            self.add_provenance(scan.output_ptr(), needle);
        }
    }
}

#[cfg(test)]
mod tests {
    test_opts! {
        scanl_terminates,
        input = [],
        output = [10],
        |graph, effect, tape_len| {
            let ptr = graph.int(Ptr::new(356, tape_len));
            let step = graph.int(Ptr::new(1, tape_len));
            let needle = graph.byte(10);

            let store = graph.store(ptr.value(), needle.value(), effect);
            let scan = graph.scanl(
                ptr.value(),
                step.value(),
                needle.value(),
                store.output_effect(),
            );
            let (scan_ptr, scan_effect) = (scan.output_ptr(), scan.output_effect());
            let load = graph.load(scan_ptr, scan_effect);

            graph.output(load.output_value(), load.output_effect()).output_effect()
        },
    }

    test_opts! {
        scanr_terminates,
        input = [],
        output = [10],
        |graph, effect, tape_len| {
            let ptr = graph.int(Ptr::new(356, tape_len));
            let step = graph.int(Ptr::new(1, tape_len));
            let needle = graph.byte(10);

            let store = graph.store(ptr.value(), needle.value(), effect);
            let scan = graph.scanr(
                ptr.value(),
                step.value(),
                needle.value(),
                store.output_effect(),
            );
            let (scan_ptr, scan_effect) = (scan.output_ptr(), scan.output_effect());
            let load = graph.load(scan_ptr, scan_effect);

            graph.output(load.output_value(), load.output_effect()).output_effect()
        },
    }
}
