use crate::{
    graph::{
        Add, Bool, Byte, Eq, Gamma, Input, InputParam, Int, Load, Mul, Neg, Neq, NodeExt, Not,
        OutputPort, Rvsdg, Sub, Theta,
    },
    ir::Const,
    passes::{
        utils::{ChangeReport, Changes, ConstantStore},
        Pass,
    },
    utils::HashMap,
    values::{Cell, Ptr},
};
use std::collections::{hash_map::Entry, BTreeMap};
use union_find::{QuickFindUf, UnionByRankSize, UnionFind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum UnionedNode {
    Const(Const),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Eq(usize, usize),
    Neq(usize, usize),
    Neg(usize),
    Not(usize),
}

/// Deduplicates constants within the graph, reusing them as much as possible
pub struct ExprDedup {
    constants: ConstantStore,
    changes: Changes<10>,
    lookup: QuickFindUf<UnionByRankSize>,
    unioned_values: HashMap<UnionedNode, usize>,
    exprs: BTreeMap<OutputPort, usize>,
    values: BTreeMap<usize, OutputPort>,
}

impl ExprDedup {
    pub fn new(tape_len: u16) -> Self {
        Self {
            constants: ConstantStore::new(tape_len),
            changes: Changes::new([
                "dedup-add",
                "dedup-sub",
                "dedup-mul",
                "dedup-eq",
                "dedup-neq",
                "dedup-neg",
                "dedup-not",
                "constants",
                "dedup-loads",
                "invariant-theta-inputs",
            ]),

            lookup: QuickFindUf::new(0),
            unioned_values: HashMap::default(),
            exprs: BTreeMap::new(),
            values: BTreeMap::new(),
        }
    }

    fn add(&mut self, output: OutputPort) -> usize {
        let id = self.lookup.insert(UnionByRankSize::default());
        self.exprs.insert(output, id);
        self.values.insert(id, output);

        id
    }
}

// TODO: Use a union-find to deduplicate all expressions
// TODO: Deduplicate invariant loop inputs
impl Pass for ExprDedup {
    fn pass_name(&self) -> &str {
        "expression-deduplication"
    }

    fn did_change(&self) -> bool {
        self.changes.did_change()
    }

    fn reset(&mut self) {
        self.constants.clear();
        self.lookup = QuickFindUf::new(0);
        self.unioned_values.clear();
        self.exprs.clear();
        self.values.clear();
        self.changes.reset();
    }

    fn report(&self) -> ChangeReport {
        self.changes.as_report()
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;
        let (mut true_visitor, mut false_visitor) = (
            Self::new(self.constants.tape_len()),
            Self::new(self.constants.tape_len()),
        );

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[true_param, false_param]) in gamma.inputs().iter().zip(gamma.input_params())
        {
            let true_param = gamma.true_branch().to_node::<InputParam>(true_param);
            let false_param = gamma.false_branch().to_node::<InputParam>(false_param);

            true_visitor.add(true_param.output());
            false_visitor.add(false_param.output());

            if let Some(constant) = self.constants.get(graph.input_source(input)) {
                true_visitor.constants.add(true_param.output(), constant);
                false_visitor.constants.add(false_param.output(), constant);
            }
        }

        changed |= true_visitor.visit_graph(gamma.true_mut());
        self.changes.combine(&true_visitor.changes);

        changed |= false_visitor.visit_graph(gamma.false_mut());
        self.changes.combine(&false_visitor.changes);

        for &output in gamma.outputs() {
            self.add(output);
        }

        if changed {
            graph.replace_node(gamma.node(), gamma);
        }
    }

    // TODO: There's some push/pull-based things we should do for routing constant values
    //       into regions so that we could avoid duplicating constant values within
    //       regions. However, this could have the downside of requiring more/better
    //       constant propagation as the values of constants wouldn't be immediately
    //       available. Everything's a tradeoff, the work involved with this one combined
    //       with its potential failure make it a low priority
    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut visitor = Self::new(self.constants.tape_len());

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (input, param) in theta.invariant_input_pairs() {
            if let Some(constant) = self.constants.get(graph.input_source(input)) {
                visitor.constants.add(param.output(), constant);
            }
        }

        for input in theta.input_params().filter(InputParam::is_value) {
            visitor.add(input.output());
        }

        let mut changed = visitor.visit_graph(theta.body_mut());
        self.changes.combine(&visitor.changes);

        // Deduplicate invariant parameters
        // TODO: Buffers
        let sources: Vec<_> = theta
            .invariant_input_pairs()
            .map(|(port, param)| (port, graph.input_source(port), param))
            .collect();
        let inputs: Vec<_> = theta.invariant_input_pairs().collect();

        for (port, param) in inputs {
            let source = graph.input_source(port);

            if theta.contains_invariant_input(port) {
                if let Some((new_port, new_source, new_param)) = sources
                    .iter()
                    .find(|&&(input, parm_source, _)| {
                        theta.contains_invariant_input(input)
                            && input != port
                            && source == parm_source
                    })
                    .copied()
                {
                    tracing::trace!(
                        old_port = ?port,
                        old_source = ?source,
                        old_param = ?param,
                        ?new_port,
                        ?new_source,
                        ?new_param,
                        ?theta,
                        "deduplicated invariant theta input {:?} with {:?}",
                        port,
                        new_port,
                    );

                    theta
                        .body_mut()
                        .rewire_dependents(param.output(), new_param.output());
                    theta.remove_invariant_input(port);
                    theta.body_mut().remove_node(param.node());

                    self.changes.inc::<"invariant-theta-inputs">();
                    changed = true;
                }
            }
        }

        for output in theta
            .output_pairs()
            .filter_map(|(port, param)| param.is_value().then_some(port))
        {
            self.add(output);
        }

        if changed {
            graph.replace_node(theta.node(), theta);
        }
    }

    fn visit_load(&mut self, graph: &mut Rvsdg, load: Load) {
        let _: Option<()> = try {
            let next_load = *graph.get_output(load.output_effect())?.0.as_load()?;

            let current_ptr = graph.input_source(load.ptr());
            let next_ptr = graph.input_source(next_load.ptr());

            // If we see a situation like
            // ```
            // x := load ptr
            // y := load ptr
            // ```
            // we can remove the second load and replace it with the first
            if current_ptr == next_ptr {
                tracing::trace!(
                    current_load = ?load,
                    next_load = ?next_load,
                    ptr = ?current_ptr,
                    "deduplicated two loads of the same address",
                );

                graph.rewire_dependents(next_load.output_value(), load.output_value());
                graph.rewire_dependents(next_load.output_effect(), load.output_effect());
                graph.remove_node(next_load.node());

                self.changes.inc::<"dedup-loads">();
            }
        };

        // Make each load unique
        self.add(load.output_value());
    }

    fn visit_input(&mut self, _graph: &mut Rvsdg, input: Input) {
        // Make each input call unique
        self.add(input.output_value());
    }

    fn visit_input_param(&mut self, _graph: &mut Rvsdg, input: InputParam) {
        // Make each input param unique
        self.add(input.output());
    }

    fn visit_add(&mut self, graph: &mut Rvsdg, add: Add) {
        let _: Option<()> = try {
            let id = self.lookup.insert(UnionByRankSize::default());

            let lhs = self
                .lookup
                .find(*self.exprs.get(&graph.input_source(add.lhs()))?);
            let rhs = self
                .lookup
                .find(*self.exprs.get(&graph.input_source(add.rhs()))?);

            match self.unioned_values.entry(UnionedNode::Add(lhs, rhs)) {
                Entry::Occupied(occupied) => {
                    let union_id = *occupied.get();
                    self.lookup.union(id, union_id);
                    self.exprs.insert(add.value(), union_id);

                    graph.rewire_dependents(add.value(), self.values[&union_id]);
                    graph.remove_outputs(add.node());

                    self.changes.inc::<"dedup-add">();
                }

                Entry::Vacant(vacant) => {
                    vacant.insert(id);
                    self.exprs.insert(add.value(), id);
                    self.values.insert(id, add.value());
                }
            }
        };
    }

    fn visit_sub(&mut self, graph: &mut Rvsdg, sub: Sub) {
        let _: Option<()> = try {
            let id = self.lookup.insert(UnionByRankSize::default());

            let lhs = self
                .lookup
                .find(*self.exprs.get(&graph.input_source(sub.lhs()))?);
            let rhs = self
                .lookup
                .find(*self.exprs.get(&graph.input_source(sub.rhs()))?);

            match self.unioned_values.entry(UnionedNode::Sub(lhs, rhs)) {
                Entry::Occupied(occupied) => {
                    let union_id = *occupied.get();
                    self.lookup.union(id, union_id);
                    self.exprs.insert(sub.value(), union_id);

                    graph.rewire_dependents(sub.value(), self.values[&union_id]);
                    graph.remove_outputs(sub.node());

                    self.changes.inc::<"dedup-sub">();
                }

                Entry::Vacant(vacant) => {
                    vacant.insert(id);
                    self.exprs.insert(sub.value(), id);
                    self.values.insert(id, sub.value());
                }
            }
        };
    }

    fn visit_mul(&mut self, graph: &mut Rvsdg, mul: Mul) {
        let _: Option<()> = try {
            let id = self.lookup.insert(UnionByRankSize::default());

            let lhs = self
                .lookup
                .find(*self.exprs.get(&graph.input_source(mul.lhs()))?);
            let rhs = self
                .lookup
                .find(*self.exprs.get(&graph.input_source(mul.rhs()))?);

            match self.unioned_values.entry(UnionedNode::Mul(lhs, rhs)) {
                Entry::Occupied(occupied) => {
                    let union_id = *occupied.get();
                    self.lookup.union(id, union_id);
                    self.exprs.insert(mul.value(), union_id);

                    graph.rewire_dependents(mul.value(), self.values[&union_id]);
                    graph.remove_outputs(mul.node());

                    self.changes.inc::<"dedup-mul">();
                }

                Entry::Vacant(vacant) => {
                    vacant.insert(id);
                    self.exprs.insert(mul.value(), id);
                    self.values.insert(id, mul.value());
                }
            }
        };
    }

    fn visit_eq(&mut self, graph: &mut Rvsdg, eq: Eq) {
        let _: Option<()> = try {
            let id = self.lookup.insert(UnionByRankSize::default());

            let lhs = self
                .lookup
                .find(*self.exprs.get(&graph.input_source(eq.lhs()))?);
            let rhs = self
                .lookup
                .find(*self.exprs.get(&graph.input_source(eq.rhs()))?);

            match self.unioned_values.entry(UnionedNode::Eq(lhs, rhs)) {
                Entry::Occupied(occupied) => {
                    let union_id = *occupied.get();
                    self.lookup.union(id, union_id);
                    self.exprs.insert(eq.value(), union_id);

                    graph.rewire_dependents(eq.value(), self.values[&union_id]);
                    graph.remove_outputs(eq.node());

                    self.changes.inc::<"dedup-eq">();
                }

                Entry::Vacant(vacant) => {
                    vacant.insert(id);
                    self.exprs.insert(eq.value(), id);
                    self.values.insert(id, eq.value());
                }
            }
        };
    }

    fn visit_neq(&mut self, graph: &mut Rvsdg, neq: Neq) {
        let _: Option<()> = try {
            let id = self.lookup.insert(UnionByRankSize::default());

            let lhs = self
                .lookup
                .find(*self.exprs.get(&graph.input_source(neq.lhs()))?);
            let rhs = self
                .lookup
                .find(*self.exprs.get(&graph.input_source(neq.rhs()))?);

            match self.unioned_values.entry(UnionedNode::Neq(lhs, rhs)) {
                Entry::Occupied(occupied) => {
                    let union_id = *occupied.get();
                    self.lookup.union(id, union_id);
                    self.exprs.insert(neq.value(), union_id);

                    graph.rewire_dependents(neq.value(), self.values[&union_id]);
                    graph.remove_outputs(neq.node());

                    self.changes.inc::<"dedup-neq">();
                }

                Entry::Vacant(vacant) => {
                    vacant.insert(id);
                    self.exprs.insert(neq.value(), id);
                    self.values.insert(id, neq.value());
                }
            }
        };
    }

    fn visit_neg(&mut self, graph: &mut Rvsdg, neg: Neg) {
        let _: Option<()> = try {
            let id = self.lookup.insert(UnionByRankSize::default());

            let input = self
                .lookup
                .find(*self.exprs.get(&graph.input_source(neg.input()))?);

            match self.unioned_values.entry(UnionedNode::Neg(input)) {
                Entry::Occupied(occupied) => {
                    let union_id = *occupied.get();
                    self.lookup.union(id, union_id);
                    self.exprs.insert(neg.value(), union_id);

                    graph.rewire_dependents(neg.value(), self.values[&union_id]);
                    graph.remove_outputs(neg.node());

                    self.changes.inc::<"dedup-neg">();
                }

                Entry::Vacant(vacant) => {
                    vacant.insert(id);
                    self.exprs.insert(neg.value(), id);
                    self.values.insert(id, neg.value());
                }
            }
        };
    }

    fn visit_not(&mut self, graph: &mut Rvsdg, not: Not) {
        let _: Option<()> = try {
            let id = self.lookup.insert(UnionByRankSize::default());

            let input = self
                .lookup
                .find(*self.exprs.get(&graph.input_source(not.input()))?);

            match self.unioned_values.entry(UnionedNode::Not(input)) {
                Entry::Occupied(occupied) => {
                    let union_id = *occupied.get();
                    self.lookup.union(id, union_id);
                    self.exprs.insert(not.value(), union_id);

                    graph.rewire_dependents(not.value(), self.values[&union_id]);
                    graph.remove_outputs(not.node());

                    self.changes.inc::<"dedup-not">();
                }

                Entry::Vacant(vacant) => {
                    vacant.insert(id);
                    self.exprs.insert(not.value(), id);
                    self.values.insert(id, not.value());
                }
            }
        };
    }

    fn visit_int(&mut self, graph: &mut Rvsdg, int: Int, value: Ptr) {
        let id = self.lookup.insert(UnionByRankSize::default());

        match self.unioned_values.entry(UnionedNode::Const(value.into())) {
            Entry::Occupied(occupied) => {
                let union_id = *occupied.get();
                self.lookup.union(union_id, id);
                self.exprs.insert(int.value(), union_id);

                graph.rewire_dependents(int.value(), self.values[&union_id]);
                graph.remove_outputs(int.node());

                self.changes.inc::<"constants">();
            }

            Entry::Vacant(vacant) => {
                vacant.insert(id);
                self.exprs.insert(int.value(), id);
                self.values.insert(id, int.value());
            }
        }

        self.constants.add(int.value(), value);
    }

    fn visit_byte(&mut self, graph: &mut Rvsdg, byte: Byte, value: Cell) {
        let id = self.lookup.insert(UnionByRankSize::default());

        match self.unioned_values.entry(UnionedNode::Const(value.into())) {
            Entry::Occupied(occupied) => {
                let union_id = *occupied.get();
                self.lookup.union(union_id, id);
                self.exprs.insert(byte.value(), union_id);

                graph.rewire_dependents(byte.value(), self.values[&union_id]);
                graph.remove_outputs(byte.node());

                self.changes.inc::<"constants">();
            }

            Entry::Vacant(vacant) => {
                vacant.insert(id);
                self.exprs.insert(byte.value(), id);
                self.values.insert(id, byte.value());
            }
        }

        self.constants.add(byte.value(), value);
    }

    fn visit_bool(&mut self, graph: &mut Rvsdg, bool: Bool, value: bool) {
        let id = self.lookup.insert(UnionByRankSize::default());

        match self.unioned_values.entry(UnionedNode::Const(value.into())) {
            Entry::Occupied(occupied) => {
                let union_id = *occupied.get();
                self.lookup.union(union_id, id);
                self.exprs.insert(bool.value(), union_id);

                graph.rewire_dependents(bool.value(), self.values[&union_id]);
                graph.remove_outputs(bool.node());

                self.changes.inc::<"constants">();
            }

            Entry::Vacant(vacant) => {
                vacant.insert(id);
                self.exprs.insert(bool.value(), id);
                self.values.insert(id, bool.value());
            }
        }

        self.constants.add(bool.value(), value);
    }
}
