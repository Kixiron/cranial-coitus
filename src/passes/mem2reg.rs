use crate::{
    graph::{
        Bool, EdgeKind, Gamma, InputParam, Int, Load, Node, NodeExt, NodeId, OutputPort, Rvsdg,
        Store, Theta,
    },
    ir::Const,
    passes::Pass,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
};

/// Evaluates constant loads within the program
pub struct Mem2Reg {
    values: BTreeMap<OutputPort, Place>,
    tape: Vec<Place>,
    changed: bool,
    buffer: Vec<&'static Node>,
    visited_buf: BTreeSet<NodeId>,
}

// TODO: Propagate port places into subgraphs by adding input ports
impl Mem2Reg {
    pub fn new(tape_len: usize) -> Self {
        Self {
            values: BTreeMap::new(),
            tape: vec![Place::Const(Const::Int(0)); tape_len],
            changed: false,
            buffer: Vec::new(),
            visited_buf: BTreeSet::new(),
        }
    }

    pub fn unknown(tape_len: usize) -> Self {
        Self {
            values: BTreeMap::new(),
            tape: vec![Place::Const(Const::Int(0)); tape_len],
            changed: false,
            buffer: Vec::new(),
            visited_buf: BTreeSet::new(),
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    fn with_buffer<'a, F, R>(&mut self, with: F) -> R
    where
        F: FnOnce(&mut Vec<&'a Node>, &mut BTreeSet<NodeId>) -> R,
    {
        let (ptr, len, cap) = Vec::into_raw_parts(mem::take(&mut self.buffer));
        // Safety: Different lifetimes are valid for transmute, see
        // https://github.com/rust-lang/unsafe-code-guidelines/issues/282
        let mut buffer: Vec<&'a Node> = unsafe { Vec::from_raw_parts(ptr.cast(), len, cap) };

        let ret = with(&mut buffer, &mut self.visited_buf);
        buffer.clear();
        self.visited_buf.clear();

        let (ptr, len, cap) = Vec::into_raw_parts(buffer);
        // Safety: Different lifetimes are valid for transmute
        self.buffer = unsafe { Vec::from_raw_parts(ptr.cast(), len, cap) };

        ret
    }
}

// TODO: Better analyze stores on the outs from thetas & gammas for fine-grained
//       tape invalidation
impl Pass for Mem2Reg {
    fn pass_name(&self) -> &str {
        "mem2reg"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.values.clear();
        self.changed = false;

        for cell in &mut self.tape {
            *cell = Place::Const(Const::Int(0));
        }
    }

    fn visit_load(&mut self, graph: &mut Rvsdg, load: Load) {
        let (ptr, source, _) = graph.get_input(load.ptr());
        let ptr = ptr
            .as_int()
            .map(|(_, ptr)| ptr)
            .or_else(|| self.values.get(&source).and_then(Place::convert_to_i32));

        if let Some(offset) = ptr {
            let offset = offset.rem_euclid(self.tape.len() as i32) as usize;

            let mut done = false;
            if let Place::Const(value) = &self.tape[offset] {
                if let Some(value) = value.convert_to_i32() {
                    tracing::debug!(
                        "replacing load from {} with value {:#04X}: {:?}",
                        offset,
                        value,
                        load,
                    );

                    let int = graph.int(value);
                    self.values.insert(int.value(), value.into());

                    graph.splice_ports(load.effect_in(), load.effect());
                    graph.rewire_dependents(load.value(), int.value());
                    graph.remove_node(load.node());

                    self.changed();
                    done = true;
                }
            }

            if !done {
                if let Place::Port(output_port) = self.tape[offset] {
                    tracing::debug!(
                        "replacing load from {} with port {:?}: {:?}",
                        offset,
                        output_port,
                        load,
                    );

                    graph.splice_ports(load.effect_in(), load.effect());
                    graph.rewire_dependents(load.value(), output_port);
                    graph.remove_node(load.node());

                    self.changed();
                }
            }
        }
    }

    fn visit_store(&mut self, graph: &mut Rvsdg, store: Store) {
        let (ptr, source, _) = graph.get_input(store.ptr());
        let ptr = ptr
            .as_int()
            .map(|(_, ptr)| ptr)
            .or_else(|| self.values.get(&source).and_then(Place::convert_to_i32));

        if let Some(offset) = ptr {
            let offset = offset.rem_euclid(self.tape.len() as i32) as usize;

            let (stored_value, output_port, _) = graph.get_input(store.value());
            let stored_value = stored_value.as_int().map(|(_, value)| value).or_else(|| {
                self.values
                    .get(&output_port)
                    .and_then(Place::convert_to_i32)
            });

            if let Some(value) = stored_value {
                // If the load's input is known but not constant, replace
                // it with a constant input. Note that we explicitly ignore
                // values that come from input ports, this is because we trust
                // other passes (namely `constant-deduplication` to propagate)
                // constants into regions
                if !graph.get_input(store.value()).0.is_int()
                    && !graph.get_input(store.value()).0.is_input_param()
                {
                    tracing::debug!("redirected {:?} to a constant of {}", store, value);

                    let int = graph.int(value);
                    self.values.insert(int.value(), (value as i32).into());

                    graph.remove_input_edges(store.value());
                    graph.add_value_edge(int.value(), store.value());

                    self.changed();
                }
            }

            match (self.tape[offset].convert_to_i32(), stored_value) {
                (Some(old), Some(new)) if old == new => {
                    tracing::debug!("removing identical store {:?}", store);

                    graph.splice_ports(store.effect_in(), store.effect());
                    graph.remove_node(store.node());

                    self.changed();
                }

                _ => {
                    self.tape[offset] = match stored_value {
                        Some(value) => value.into(),
                        None => Place::Port(output_port),
                    };
                }
            }
        } else {
            tracing::debug!("unknown store {:?}, invalidating tape", store);

            // Invalidate the whole tape
            for cell in self.tape.iter_mut() {
                *cell = Place::Unknown;
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

                    graph.splice_ports(load.effect_in(), load.effect());
                    graph.rewire_dependents(load.value(), graph.input_source(store.value()));
                    graph.remove_node(load.node());

                    self.changed();
                }
            }
        }
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        let replaced = self.values.insert(bool.value(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(value.into()));
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: i32) {
        let replaced = self.values.insert(int.value(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(value.into()));
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        // Don't propagate port places into subgraphs
        // FIXME: This is only a implementation limit right now,
        //        gamma nodes aren't iterative so they can have full
        //        access to the prior scope via input ports. We should
        //        add inputs as needed to bring branch-invariant code
        //        into said branches
        let tape: Vec<_> = self
            .tape
            .iter()
            .map(|place| match place {
                Place::Const(constant) => Place::Const(constant.clone()),
                _ => Place::Unknown,
            })
            .collect();

        // Both branches of the gamma node get the previous context, the changes they
        // create within it just are trickier to propagate
        let (mut truthy_visitor, mut falsy_visitor) = (
            Self {
                tape: tape.clone(),
                ..Self::unknown(self.tape.len())
            },
            Self {
                tape,
                ..Self::unknown(self.tape.len())
            },
        );

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[truthy_param, falsy_param]) in
            gamma.inputs().iter().zip(gamma.input_params())
        {
            if let Some(constant) = self
                .values
                .get(&graph.input_source(input))
                .filter(|place| place.is_const())
                .cloned()
            {
                let param = gamma.true_branch().to_node::<InputParam>(truthy_param);
                let replaced = truthy_visitor
                    .values
                    .insert(param.output(), constant.clone());
                debug_assert!(replaced.is_none());

                let param = gamma.false_branch().to_node::<InputParam>(falsy_param);
                let replaced = falsy_visitor.values.insert(param.output(), constant);
                debug_assert!(replaced.is_none());
            }
        }

        // TODO: Eliminate gamma branches based on gamma condition

        truthy_visitor.visit_graph(gamma.true_mut());
        falsy_visitor.visit_graph(gamma.false_mut());
        self.changed |= truthy_visitor.did_change();
        self.changed |= falsy_visitor.did_change();

        // TODO: Propagate constants out of gamma bodies?

        // Figure out if there's any stores within either of the gamma branches
        let (truthy_stores, falsy_stores) = self.with_buffer(|buffer, visited| {
            gamma.true_branch().transitive_nodes_into(buffer, visited);
            visited.clear();
            let truthy_stores = buffer.drain(..).filter(|node| node.is_store()).count();

            gamma.false_branch().transitive_nodes_into(buffer, visited);
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
                "gamma node does {} stores ({} in true branch, {} in false branch), invalidating program tape",
                truthy_stores + falsy_stores,
                truthy_stores,
                falsy_stores,
            );

            for cell in self.tape.iter_mut() {
                *cell = Place::Unknown;
            }
        } else {
            tracing::debug!("gamma node does no stores, not invalidating program tape");
        }

        graph.replace_node(gamma.node(), gamma);
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let body_stores = self.with_buffer(|buffer, visited| {
            theta.body().transitive_nodes_into(buffer, visited);
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
            vec![Place::Unknown; self.tape.len()]
        } else {
            self.tape
                .iter()
                .map(|place| match place {
                    Place::Const(constant) => Place::Const(constant.clone()),
                    _ => Place::Unknown,
                })
                .collect()
        };

        let mut visitor = Self {
            tape,
            ..Self::unknown(self.tape.len())
        };

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (input, param) in theta.input_pairs() {
            if let Some(constant) = self
                .values
                .get(&graph.input_source(input))
                .filter(|place| place.is_const())
                .cloned()
            {
                if !constant.is_port() {
                    let replaced = visitor.values.insert(param.output(), constant);
                    debug_assert!(replaced.is_none());
                }
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
                *cell = Place::Unknown;
            }
        } else {
            tracing::debug!("theta body does no stores, not invalidating program tape");
        }

        graph.replace_node(theta.node(), theta);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Place {
    Unknown,
    Const(Const),
    Port(OutputPort),
}

impl Place {
    pub fn convert_to_i32(&self) -> Option<i32> {
        if let Self::Const(constant) = self {
            constant.as_int()
        } else {
            None
        }
    }

    /// Returns `true` if the place is [`Const`].
    ///
    /// [`Const`]: Place::Const
    pub const fn is_const(&self) -> bool {
        matches!(self, Self::Const(..))
    }

    /// Returns `true` if the place is [`Port`].
    ///
    /// [`Port`]: Place::Port
    pub const fn is_port(&self) -> bool {
        matches!(self, Self::Port(..))
    }
}

impl From<i32> for Place {
    fn from(int: i32) -> Self {
        Self::Const(Const::Int(int))
    }
}

impl From<bool> for Place {
    fn from(bool: bool) -> Self {
        Self::Const(Const::Bool(bool))
    }
}

impl From<OutputPort> for Place {
    fn from(port: OutputPort) -> Self {
        Self::Port(port)
    }
}
