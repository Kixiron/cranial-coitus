use crate::{
    graph::{
        Bool, Byte, EdgeKind, Gamma, InputParam, Int, Load, Node, NodeExt, OutputPort, Rvsdg,
        Store, Theta,
    },
    ir::Const,
    passes::{utils::ChangeReport, Pass},
    utils::AssertNone,
    values::{Cell, Ptr},
};
use std::collections::BTreeMap;

/// Evaluates constant loads within the program
pub struct Mem2Reg {
    // FIXME: ConstantStore
    values: BTreeMap<OutputPort, Place>,
    tape: Vec<Place>,
    tape_len: u16,
    changed: bool,
    constant_loads_elided: usize,
    loads_elided: usize,
    constant_stores_elided: usize,
    identical_stores_removed: usize,
    dependent_loads_removed: usize,
}

// TODO: Propagate port places into subgraphs by adding input ports
impl Mem2Reg {
    pub fn new(tape_len: u16) -> Self {
        Self {
            values: BTreeMap::new(),
            tape: vec![Place::Const(Const::Cell(Cell::zero())); tape_len as usize],
            tape_len,
            changed: false,
            constant_loads_elided: 0,
            loads_elided: 0,
            constant_stores_elided: 0,
            identical_stores_removed: 0,
            dependent_loads_removed: 0,
        }
    }

    pub fn unknown(tape_len: u16) -> Self {
        Self {
            values: BTreeMap::new(),
            tape: vec![Place::Unknown; tape_len as usize],
            tape_len,
            changed: false,
            constant_loads_elided: 0,
            loads_elided: 0,
            constant_stores_elided: 0,
            identical_stores_removed: 0,
            dependent_loads_removed: 0,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    fn update_counts(&mut self, other: &Self) {
        self.constant_loads_elided += other.constant_loads_elided;
        self.loads_elided += other.loads_elided;
        self.constant_stores_elided += other.constant_stores_elided;
        self.identical_stores_removed += other.identical_stores_removed;
        self.dependent_loads_removed += other.dependent_loads_removed;
    }
}

// TODO: Better analyze stores on the outs from thetas & gammas for fine-grained
//       tape invalidation
impl Pass for Mem2Reg {
    fn pass_name(&self) -> &'static str {
        "mem2reg"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn report(&self) -> ChangeReport {
        map! {
            "constant loads" => self.constant_loads_elided,
            "loads" => self.loads_elided,
            "dependent loads" => self.dependent_loads_removed,
            "constant stores" => self.constant_stores_elided,
            "identical stores" => self.identical_stores_removed,
        }
    }

    fn reset(&mut self) {
        self.values.clear();
        self.changed = false;

        for cell in &mut self.tape {
            *cell = Place::Const(Const::Cell(Cell::zero()));
        }
    }

    fn visit_load(&mut self, graph: &mut Rvsdg, load: Load) {
        let (ptr, source, _) = graph.get_input(load.ptr());
        // FIXME: Bytes
        let ptr = ptr.as_int_value().or_else(|| {
            self.values
                .get(&source)
                .and_then(|place| place.as_ptr(self.tape_len))
        });

        if let Some(offset) = ptr {
            let mut done = false;
            if let Place::Const(value) = &self.tape[offset] {
                let value = value.into_ptr(self.tape_len);
                tracing::debug!(
                    "replacing load from {} with value {:#04X}: {:?}",
                    offset,
                    value,
                    load,
                );

                let int = graph.int(value);
                self.values.insert(int.value(), value.into());

                graph.splice_ports(load.input_effect(), load.output_effect());
                graph.rewire_dependents(load.output_value(), int.value());
                graph.remove_node(load.node());

                self.changed();
                done = true;
                self.constant_loads_elided += 1;
            }

            if !done {
                if let Place::Port(output_port) = self.tape[offset] {
                    tracing::debug!(
                        "replacing load from {} with port {:?}: {:?}",
                        offset,
                        output_port,
                        load,
                    );

                    graph.splice_ports(load.input_effect(), load.output_effect());
                    graph.rewire_dependents(load.output_value(), output_port);
                    graph.remove_node(load.node());

                    self.changed();
                    self.loads_elided += 1;
                }
            }
        }
    }

    fn visit_store(&mut self, graph: &mut Rvsdg, store: Store) {
        let (ptr, source, _) = graph.get_input(store.ptr());
        // FIXME: Bytes
        let ptr = ptr.as_int_value().or_else(|| {
            self.values
                .get(&source)
                .and_then(|place| place.as_ptr(self.tape_len))
        });

        if let Some(offset) = ptr {
            let (stored_value, output_port, _) = graph.get_input(store.value());
            // FIXME: Bytes
            let stored_value = stored_value.as_int_value().or_else(|| {
                self.values
                    .get(&output_port)
                    .and_then(|place| place.as_ptr(self.tape_len))
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
                    self.values.insert(int.value(), value.into());

                    graph.remove_input_edges(store.value());
                    graph.add_value_edge(int.value(), store.value());

                    self.changed();
                    self.constant_stores_elided += 1;
                }
            }

            match (self.tape[offset].as_ptr(self.tape_len), stored_value) {
                (Some(old), Some(new)) if old == new => {
                    tracing::debug!("removing identical store {:?}", store);

                    graph.splice_ports(store.effect_in(), store.output_effect());
                    graph.remove_node(store.node());

                    self.changed();
                    self.identical_stores_removed += 1;
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

        let effect_output = graph.get_output(store.output_effect());
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
                if graph.input_source_id(load.ptr()) == graph.input_source_id(store.ptr()) {
                    tracing::debug!(
                        "replaced dependent load with value {:?} (store: {:?})",
                        load,
                        store,
                    );

                    graph.splice_ports(load.input_effect(), load.output_effect());
                    graph.rewire_dependents(load.output_value(), graph.input_source(store.value()));
                    graph.remove_node(load.node());

                    self.changed();
                    self.dependent_loads_removed += 1;
                }
            }
        }
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        let replaced = self.values.insert(bool.value(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(value.into()));
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: Ptr) {
        let replaced = self.values.insert(int.value(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(value.into()));
    }

    fn visit_byte(&mut self, _graph: &mut Rvsdg, byte: Byte, value: Cell) {
        self.values.insert(byte.value(), value.into());
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;

        // Don't propagate port places into subgraphs
        // FIXME: This is only a implementation limit right now,
        //        gamma nodes aren't iterative so they can have full
        //        access to the prior scope via input ports. We should
        //        add inputs as needed to bring branch-invariant code
        //        into said branches
        let tape: Vec<_> = self
            .tape
            .iter()
            .map(|place| match *place {
                Place::Const(constant) => Place::Const(constant),
                _ => Place::Unknown,
            })
            .collect();

        // Both branches of the gamma node get the previous context, the changes they
        // create within it just are trickier to propagate
        let (mut true_visitor, mut false_visitor) = (
            Self {
                tape: tape.clone(),
                ..Self::unknown(self.tape_len)
            },
            Self {
                tape,
                ..Self::unknown(self.tape_len)
            },
        );

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        let inputs: Vec<_> = gamma
            .inputs()
            .iter()
            .zip(gamma.input_params())
            .map(|(&input, &params)| (input, params))
            .collect();
        for (input, [true_param, false_param]) in inputs {
            if let Some(place) = self.values.get(&graph.input_source(input)).cloned() {
                let param = gamma.true_branch().to_node::<InputParam>(true_param);
                true_visitor
                    .values
                    .insert(param.output(), place.clone())
                    .debug_unwrap_none();

                let param = gamma.false_branch().to_node::<InputParam>(false_param);
                false_visitor
                    .values
                    .insert(param.output(), place)
                    .debug_unwrap_none();
            }
        }

        changed |= true_visitor.visit_graph(gamma.true_mut());
        self.update_counts(&true_visitor);

        changed |= false_visitor.visit_graph(gamma.false_mut());
        self.update_counts(&false_visitor);

        // Figure out if there's any stores within either of the gamma branches
        let mut truthy_stores = 0;
        gamma
            .true_branch()
            .for_each_transitive_node(|_node_id, node| truthy_stores += node.is_store() as usize);

        let mut falsy_stores = 0;
        gamma
            .false_branch()
            .for_each_transitive_node(|_node_id, node| falsy_stores += node.is_store() as usize);

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

        if changed {
            graph.replace_node(gamma.node(), gamma);
            self.changed();
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut changed = false;
        let mut body_stores = 0;
        theta
            .body()
            .for_each_transitive_node(|_node_id, node| body_stores += node.is_store() as usize);

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
                .map(|place| match *place {
                    Place::Const(constant) => Place::Const(constant),
                    _ => Place::Unknown,
                })
                .collect()
        };

        let mut visitor = Self {
            tape,
            ..Self::unknown(self.tape_len)
        };

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        // Note: We only propagate **invariant** inputs into the loop, propagating
        //       variant inputs requires dataflow information
        for (input, param) in theta.invariant_input_pairs() {
            if let Some(constant) = self
                .values
                .get(&graph.input_source(input))
                .filter(|place| place.is_const())
                .cloned()
            {
                visitor
                    .values
                    .insert(param.output(), constant)
                    .debug_unwrap_none();
            }
        }

        changed |= visitor.visit_graph(theta.body_mut());
        self.update_counts(&visitor);

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

        if changed {
            graph.replace_node(theta.node(), theta);
            self.changed();
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Place {
    Unknown,
    Const(Const),
    Port(OutputPort),
}

impl Place {
    pub fn as_ptr(&self, tape_len: u16) -> Option<Ptr> {
        if let Self::Const(constant) = self {
            Some(constant.into_ptr(tape_len))
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
}

impl From<Ptr> for Place {
    fn from(ptr: Ptr) -> Self {
        Self::Const(Const::Ptr(ptr))
    }
}

impl From<Cell> for Place {
    fn from(cell: Cell) -> Self {
        Self::Const(Const::Cell(cell))
    }
}

impl From<u8> for Place {
    fn from(byte: u8) -> Self {
        Self::Const(Const::Cell(Cell::new(byte)))
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
