mod arithmetic;
mod domain;
mod equality;
mod gamma;
mod memory;
mod pass;
mod theta;

use crate::{
    graph::{InputParam, InputPort, NodeId, OutputPort, Rvsdg},
    passes::utils::Changes,
    utils::{AssertNone, ImHashMap},
};
use domain::{ByteSet, Domain};
use im_rc::{hashmap::HashMapPool, vector::RRBPool, Vector};
use std::{fmt::Debug, rc::Rc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataflowSettings {
    /// The length of the program tape
    tape_len: u16,

    /// Whether or not the tape pointer can overflow and underflow
    tape_operations_wrap: bool,

    /// Whether or not cells can overflow and underflow
    cell_operations_wrap: bool,
}

impl DataflowSettings {
    pub const fn new(
        tape_len: u16,
        tape_operations_wrap: bool,
        cell_operations_wrap: bool,
    ) -> Self {
        Self {
            tape_len,
            tape_operations_wrap,
            cell_operations_wrap,
        }
    }
}

pub struct Dataflow {
    changes: Changes<6>,
    values: ImHashMap<OutputPort, Domain>,
    constraints: ImHashMap<(OutputPort, OutputPort), (Domain, Domain)>,
    // TODO: Memory location constraints, constraints that pointers have on
    //       the cell they point to
    tape: Vector<ByteSet>,
    settings: DataflowSettings,
    can_mutate: bool,
}

impl Dataflow {
    pub fn new(settings: DataflowSettings) -> Self {
        let (value_pool, constraint_pool, tape_pool) = (
            HashMapPool::new(1024),
            HashMapPool::new(256),
            RRBPool::new(settings.tape_len as usize),
        );
        let values = ImHashMap::with_pool_hasher(&value_pool, Rc::new(Default::default()));
        let constraints =
            ImHashMap::with_pool_hasher(&constraint_pool, Rc::new(Default::default()));

        // Zero-initialize the program tape
        let mut tape = Vector::with_pool(&tape_pool);
        tape.extend((0..settings.tape_len).map(|_| ByteSet::singleton(0)));

        Self {
            changes: Self::new_changes(),
            values,
            constraints,
            tape,
            settings,
            can_mutate: true,
        }
    }

    fn clone_for_subscope(&self, values: ImHashMap<OutputPort, Domain>) -> Self {
        Self {
            changes: Self::new_changes(),
            values,
            tape: self.tape.clone(),
            constraints: self.constraints.clone(),
            settings: self.settings,
            can_mutate: false,
        }
    }

    fn allow_mutation(&mut self) {
        self.can_mutate = true;
    }

    fn with_mutation(mut self, can_mutate: bool) -> Self {
        self.can_mutate = can_mutate;
        self
    }

    fn new_changes() -> Changes<6> {
        Changes::new([
            "const-eq",
            "const-neq",
            "const-add",
            "const-sub",
            "const-load",
            "gamma-branch-elision",
        ])
    }

    const fn tape_len(&self) -> u16 {
        self.settings.tape_len
    }

    #[track_caller]
    fn add_domain<C>(&mut self, port: OutputPort, value: C)
    where
        C: Into<Domain>,
    {
        let mut value = value.into();
        self.values
            .entry(port)
            .and_modify(|domain| domain.union_mut(&mut value))
            .or_insert(value);
    }

    fn domain(&self, port: OutputPort) -> Option<&Domain> {
        self.values.get(&port)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_constraints(
        &mut self,
        comparison: OutputPort,
        value: OutputPort,
        mut true_domain: Domain,
        mut false_domain: Domain,
    ) {
        self.constraints
            .entry((comparison, value))
            .and_modify(|(first, second)| {
                first.union_mut(&mut true_domain);
                second.union_mut(&mut false_domain);
            })
            .or_insert((true_domain, false_domain));
    }

    fn collect_subgraph_inputs(
        &mut self,
        graph: &Rvsdg,
        branch_graph: &Rvsdg,
        input_pairs: impl Iterator<Item = (InputPort, NodeId)>,
    ) -> ImHashMap<OutputPort, Domain> {
        let mut values =
            ImHashMap::with_pool_hasher(self.values.pool(), Rc::clone(self.values.hasher()));

        for (input, param) in input_pairs {
            let input_src = graph.input_source(input);

            if let Some(domain) = self.domain(input_src) {
                let param = branch_graph.cast_node::<InputParam>(param).unwrap();
                values
                    .insert(param.output(), domain.clone())
                    .debug_unwrap_none();
            }
        }

        values
    }
}
