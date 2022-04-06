use crate::{
    graph::{Load, Rvsdg, Scan, ScanDirection, Store},
    passes::dataflow::{
        domain::{ByteSet, Domain, IntSet},
        Dataflow,
    },
    utils::HashSet,
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
            (None, None) => self.tape.iter_mut().for_each(|cell| cell.fill()),

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
                    self.tape[ptr].fill();
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
                    let cell = &mut self.tape[ptr];
                    cell.clear();

                    *cell = value;
                }
            }
        }

        // TODO: Finer-grained provenance invalidation
        if self.can_mutate {
            self.port_provenance.clear();
        }
    }

    pub(super) fn compute_load(&mut self, graph: &mut Rvsdg, load: Load) {
        let ptr = graph.input_source(load.ptr());
        let ptr_domain = self.values.get(&ptr);

        let mut loaded = match ptr_domain {
            Some(ptr) => {
                let ptr = ptr.into_int_set(self.settings.tape_len);
                if ptr.is_empty() {
                    tracing::error!(?load, %ptr, "loaded from pointer without any possible values?");
                    ByteSet::full()
                } else {
                    let mut ptr_values = ptr.iter();
                    let mut loaded = self.tape[ptr_values.next().unwrap()];

                    // Make the loaded value the union of all cells we
                    // could possibly load
                    for ptr in ptr_values {
                        loaded.union(self.tape[ptr]);
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
            // If there's provenance for the given pointer, the loaded value
            // will be the intersection of both the provenance and the value
            // of the pointed to cells
            if let Some(provenance) = self.provenance(ptr) {
                loaded = loaded.intersection(provenance);
            }

            if let Some(loaded) = loaded.as_singleton() {
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
        let _: Option<()> = try {
            let (ptr, step, needle) = (
                graph.input_source(scan.ptr()),
                graph.input_source(scan.step()),
                graph.input_source(scan.needle()),
            );

            let (ptr, step, needle) = (
                self.domain(ptr)
                    .map(|domain| domain.into_int_set(self.tape_len()))?,
                self.domain(step)
                    .map(|domain| domain.into_int_set(self.tape_len()))?,
                self.domain(needle).map(Domain::into_byte_set)?,
            );

            // This algorithm is pretty naive, we basically compute a cartesian product
            // of all three variables
            let (mut possible_values, mut has_matched) = (IntSet::empty(self.tape_len()), false);

            if let Some(step) = step.as_singleton() && let Some(mut ptr) = ptr.as_singleton() {
                let mut visited_cells = HashSet::default();

                loop {
                    if self.tape[ptr].intersects(&needle) {
                        possible_values.add(ptr);

                        if !has_matched
                            && let Some(value) = self.tape[ptr].as_singleton()
                            && let Some(needle) = needle.as_singleton()
                            && value == needle
                        {
                            self.add_domain(scan.output_ptr(), possible_values);
                            return;
                        }

                        has_matched = true;
                    }

                    if self.settings.tape_operations_wrap {
                        ptr = match scan.direction() {
                            ScanDirection::Forward => ptr.wrapping_add(step),
                            ScanDirection::Backward => ptr.wrapping_sub(step),
                        };

                        if !visited_cells.insert(ptr) {
                            break;
                        }
                    } else {
                        let next = match scan.direction() {
                            ScanDirection::Forward => ptr.checked_add(step),
                            ScanDirection::Backward => ptr.checked_sub(step),
                        };

                        match next {
                            Some(next) => ptr = next,
                            None => break,
                        }
                    }
                }

            // TODO: Handle this
            } else {
                // Add provenance for the scan's output
                if self.can_mutate && !needle.is_full() && !needle.is_empty() {
                    self.add_provenance(scan.output_ptr(), needle.to_owned());
                }
                return;
            }

            // TODO: What situations is this empty in?
            if !possible_values.is_empty() {
                self.add_domain(scan.output_ptr(), possible_values);
            }

            // Add provenance for the scan's output
            if self.can_mutate && !needle.is_full() && !needle.is_empty() {
                self.add_provenance(scan.output_ptr(), needle.to_owned());
            }
        };
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
