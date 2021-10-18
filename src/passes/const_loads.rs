use crate::{
    graph::{Bool, EdgeKind, Int, Load, Node, NodeId, Phi, Rvsdg, Store, Theta},
    ir::Const,
    passes::Pass,
};
use std::{collections::HashMap, mem};

/// Evaluates constant loads within the program
pub struct ConstLoads {
    values: HashMap<NodeId, Const>,
    tape: Vec<Option<i32>>,
    changed: bool,
    buffer: Vec<&'static Node>,
}

impl ConstLoads {
    pub fn new(tape_len: usize) -> Self {
        Self {
            values: HashMap::new(),
            tape: vec![Some(0); tape_len],
            changed: false,
            buffer: Vec::new(),
        }
    }

    pub fn unknown(tape_len: usize) -> Self {
        Self {
            values: HashMap::new(),
            tape: vec![None; tape_len],
            changed: false,
            buffer: Vec::new(),
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    fn with_buffer<'a, F, R>(&mut self, with: F) -> R
    where
        F: FnOnce(&mut Vec<&'a Node>) -> R,
    {
        let (ptr, len, cap) = Vec::into_raw_parts(mem::take(&mut self.buffer));
        // Safety: Different lifetimes are valid for transmute, see
        // https://github.com/rust-lang/unsafe-code-guidelines/issues/282
        let mut buffer: Vec<&'a Node> = unsafe { Vec::from_raw_parts(ptr.cast(), len, cap) };

        let ret = with(&mut buffer);
        buffer.clear();

        let (ptr, len, cap) = Vec::into_raw_parts(buffer);
        // Safety: Different lifetimes are valid for transmute
        self.buffer = unsafe { Vec::from_raw_parts(ptr.cast(), len, cap) };

        ret
    }
}

impl Pass for ConstLoads {
    fn pass_name(&self) -> &str {
        "constant-loads"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.values.clear();
        self.changed = false;

        for cell in &mut self.tape {
            *cell = Some(0);
        }
    }

    fn visit_load(&mut self, graph: &mut Rvsdg, load: Load) {
        let ptr = graph.get_input(load.ptr()).0;
        let ptr = ptr
            .as_int()
            .map(|(_, ptr)| ptr)
            .or_else(|| self.values.get(&ptr.node_id()).and_then(Const::as_int));

        if let Some(offset) = ptr {
            let offset = offset.rem_euclid(self.tape.len() as i32) as usize;

            if let Some(value) = self.tape[offset] {
                tracing::debug!(
                    "replacing load at {} with value {:#04X}: {:?}",
                    offset,
                    value,
                    load,
                );

                let int = graph.int(value);
                self.values.insert(int.node(), (value as i32).into());

                tracing::debug!("created int node {:?} with value {}", int.node(), value);
                tracing::debug!(
                    "rewiring effect edge bridging from {:?}-{:?} into {:?}->{:?} ({:?} to {:?})",
                    load.effect_in(),
                    load.effect(),
                    graph.get_input(load.effect_in()).1,
                    graph.get_output(load.effect()).unwrap().1,
                    graph.get_input(load.effect_in()).0.node_id(),
                    graph.get_output(load.effect()).unwrap().0.node_id(),
                );

                graph.splice_ports(load.effect_in(), load.effect());
                graph.rewire_dependents(load.value(), int.value());
                graph.remove_node(load.node());

                self.changed();
            }
        }
    }

    fn visit_store(&mut self, graph: &mut Rvsdg, store: Store) {
        let ptr = graph.get_input(store.ptr()).0;
        let ptr = ptr.as_int().map(|(_, ptr)| ptr).or_else(|| {
            self.values
                .get(&ptr.node_id())
                .and_then(Const::convert_to_i32)
        });

        if let Some(offset) = ptr {
            let offset = offset.rem_euclid(self.tape.len() as i32) as usize;

            let stored_value = graph.get_input(store.value()).0;
            let stored_value = stored_value.as_int().map(|(_, value)| value).or_else(|| {
                self.values
                    .get(&stored_value.node_id())
                    .and_then(Const::convert_to_i32)
            });

            if let Some(value) = stored_value {
                self.values.insert(store.node(), (value as i32).into());

                // If the load's input is known but not constant, replace
                // it with a constant input
                if !graph.get_input(store.value()).0.is_int() {
                    tracing::debug!("redirected {:?} to a constant of {}", store, value);

                    let int = graph.int(value);
                    self.values.insert(int.node(), (value as i32).into());

                    graph.remove_input(store.value());
                    graph.add_value_edge(int.value(), store.value());

                    self.changed();
                }
            }

            self.tape[offset] = stored_value;
        } else {
            tracing::debug!("unknown store {:?}, invalidating tape", store);

            // Invalidate the whole tape
            for cell in self.tape.iter_mut() {
                *cell = None;
            }
        }

        let effect_output = graph.get_output(store.effect());
        if let Some((node, _, kind)) = effect_output {
            debug_assert_eq!(kind, EdgeKind::Effect);

            // TODO: Move this to separate pass
            if let Node::Load(load) = node.clone() {
                // If there's a sequence of
                //
                // ```
                // store _0, _1
                // _2 = load _0
                // ```
                //
                // we want to remove the redundant load since nothing has occurred
                // change the pointed-to value, changing this code to
                //
                // ```
                // store _0, _1
                // _2 = _1
                // ```
                //
                // but we don't have any sort of passthrough node currently (may
                // want to fix that, could be useful) we instead rewire all dependents on
                // the redundant load (`_2`) to instead point to the known value of the cell
                // (`_1`) transforming this code
                //
                // ```
                // store _0, _1
                // _2 = load _0
                // _3 = add _2, int 10
                // ```
                //
                // into this code
                //
                // ```
                // store _0, _1
                // _3 = add _1, int 10
                // ```
                if graph.get_input(load.ptr()).0.node_id()
                    == graph.get_input(store.ptr()).0.node_id()
                {
                    tracing::debug!(
                        "replaced dependent load with value {:?} (store: {:?})",
                        load,
                        store,
                    );

                    // graph.splice_ports(load.effect_in(), load.effect());
                    // graph.rewire_dependents(load.value(), graph.input_source(store.value()));
                    // graph.remove_node(load.node());
                    //
                    // self.changed();
                }
            }
        }
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        let replaced = self.values.insert(bool.node(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Bool(value)));
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: i32) {
        let replaced = self.values.insert(int.node(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Int(value)));
    }

    fn visit_phi(&mut self, graph: &mut Rvsdg, mut phi: Phi) {
        // Both branches of the phi node get the previous context, the changes they
        // create within it just are trickier to propagate
        let (mut truthy_visitor, mut falsy_visitor) = (
            Self {
                tape: self.tape.clone(),
                ..Self::unknown(self.tape.len())
            },
            Self {
                tape: self.tape.clone(),
                ..Self::unknown(self.tape.len())
            },
        );

        // For each input into the phi region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[truthy_param, falsy_param]) in phi.inputs().iter().zip(phi.input_params()) {
            let (input_node, _, _) = graph.get_input(input);
            let input_node_id = input_node.node_id();

            if let Some(constant) = self.values.get(&input_node_id).cloned() {
                let replaced = truthy_visitor.values.insert(truthy_param, constant.clone());
                debug_assert!(replaced.is_none());

                let replaced = falsy_visitor.values.insert(falsy_param, constant);
                debug_assert!(replaced.is_none());
            }
        }

        // TODO: Eliminate phi branches based on phi condition

        truthy_visitor.visit_graph(phi.truthy_mut());
        falsy_visitor.visit_graph(phi.falsy_mut());
        self.changed |= truthy_visitor.did_change();
        self.changed |= falsy_visitor.did_change();

        // TODO: Propagate constants out of phi bodies?

        // Figure out if there's any stores within either of the phi branches
        let (truthy_stores, falsy_stores) = self.with_buffer(|buffer| {
            phi.truthy().transitive_nodes_into(buffer);
            let truthy_stores = buffer.drain(..).filter(|node| node.is_store()).count();

            phi.falsy().transitive_nodes_into(buffer);
            let falsy_stores = buffer.drain(..).filter(|node| node.is_store()).count();

            (truthy_stores, falsy_stores)
        });

        // Invalidate the whole tape if any stores occur within it
        // FIXME: We pessimistically clear out the *entire* tape, but
        //        ideally we'd only clear out the cells that are specifically
        //        stored to and only fall back to the pessimistic case in the
        //        event that we find a store to an unknown pointer
        if truthy_stores != 0 || falsy_stores != 0 {
            tracing::debug!(
                "phi node does {} stores ({} in true branch, {} in false branch), invalidating program tape",
                truthy_stores + falsy_stores,
                truthy_stores,
                falsy_stores,
            );

            for cell in self.tape.iter_mut() {
                *cell = None;
            }
        } else {
            tracing::debug!("phi node does no stores, not invalidating program tape");
        }

        graph.replace_node(phi.node(), phi);
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let body_stores = self.with_buffer(|buffer| {
            theta.body().transitive_nodes_into(buffer);
            buffer.drain(..).filter(|node| node.is_store()).count()
        });

        // If no stores occur within the theta's body then we can
        // propagate the current state of the tape into it and if not,
        // we can clear it out
        // FIXME: We pessimistically clear out the *entire* tape, but
        //        ideally we'd only clear out the cells that are specifically
        //        stored to and only fall back to the pessimistic case in the
        //        event that we find a store to an unknown pointer
        let tape = if body_stores != 0 {
            vec![None; self.tape.len()]
        } else {
            self.tape.clone()
        };

        let mut visitor = Self {
            tape,
            ..Self::unknown(self.tape.len())
        };

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &param) in theta.inputs().iter().zip(theta.input_params()) {
            let (input_node, _, _) = graph.get_input(input);
            let input_node_id = input_node.node_id();

            if let Some(constant) = self.values.get(&input_node_id).cloned() {
                let replaced = visitor.values.insert(param, constant);
                debug_assert!(replaced.is_none());
            }
        }

        visitor.visit_graph(theta.body_mut());
        self.changed |= visitor.did_change();

        // TODO: Propagate constants out of theta bodies?

        // If any stores occur within the theta's body, invalidate the whole tape
        // FIXME: We pessimistically clear out the *entire* tape, but
        //        ideally we'd only clear out the cells that are specifically
        //        stored to and only fall back to the pessimistic case in the
        //        event that we find a store to an unknown pointer
        if body_stores != 0 {
            tracing::debug!(
                "theta body does {} stores, invalidating program tape",
                body_stores,
            );

            for cell in self.tape.iter_mut() {
                *cell = None;
            }
        } else {
            tracing::debug!("theta body does no stores, not invalidating program tape");
        }

        graph.replace_node(theta.node(), theta);
    }
}
