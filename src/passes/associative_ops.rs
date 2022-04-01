use crate::{
    graph::{
        Add, Byte, Gamma, InputParam, InputPort, Int, Mul, Node, NodeExt, NodeId, Rvsdg, Theta,
    },
    ir::Const,
    passes::{
        utils::{BinOp, ChangeReport, Changes, ConstantStore},
        Pass,
    },
    utils::HashSet,
    values::{Cell, Ptr},
};

/// Fuses chained additions based on the law of associative addition
// TODO: Equality is also associative but it's unclear whether or not
//       that situation can actually arise within brainfuck programs
pub struct AssociativeOps {
    constants: ConstantStore,
    to_be_removed: HashSet<NodeId>,
    changes: Changes<2>,
    tape_len: u16,
}

impl AssociativeOps {
    pub fn new(tape_len: u16) -> Self {
        Self {
            constants: ConstantStore::new(tape_len),
            to_be_removed: HashSet::with_hasher(Default::default()),
            changes: Changes::new(["add", "mul"]),
            tape_len,
        }
    }

    fn operand(&self, graph: &Rvsdg, input: InputPort) -> (InputPort, Option<Const>) {
        let (operand, output, _) = graph.get_input(input);
        let value = operand
            .as_int_value()
            .map(Const::Ptr)
            .or_else(|| operand.as_byte_value().map(Const::Cell))
            .or_else(|| self.constants.ptr(output).map(Const::Ptr));

        (input, value)
    }

    fn fold_associative_operation<T>(&mut self, graph: &mut Rvsdg, operation: T) -> bool
    where
        T: BinOp,
        for<'a> &'a Node: TryInto<&'a T>,
    {
        let inputs = [
            self.operand(graph, operation.lhs()),
            self.operand(graph, operation.rhs()),
        ];

        // If one operand is constant and one is an un-evaluated expression
        if let [(_, Some(known)), (unknown, None)] | [(unknown, None), (_, Some(known))] = inputs {
            tracing::debug!(
                "found {} ({:?}: {:?} + {:?})",
                T::name(),
                operation.node(),
                unknown,
                known
            );

            // If the un-evaluated expression is an op node
            if let Some(dependency_operation) = graph.cast_source::<T>(unknown) {
                // Get the inputs of the dependency add
                let dependency_inputs = [
                    self.operand(graph, dependency_operation.lhs()),
                    self.operand(graph, dependency_operation.rhs()),
                ];

                // If one of the dependency add's inputs are unevaluated and one is constant
                if let [(_, Some(dependency_known)), (dependency_unknown, None)]
                | [(dependency_unknown, None), (_, Some(dependency_known))] = dependency_inputs
                {
                    let unknown_parent = graph.port_parent(graph.input_source(unknown));
                    let unknown_dependency_parent =
                        graph.port_parent(graph.input_source(dependency_unknown));

                    tracing::debug!(
                        "found dependency of {} ({:?}: {:?}->{:?} {op} {:?}), fusing it with ({:?}: {:?}->{:?} {op} {:?})",
                        T::name(),
                        operation.node(),
                        unknown,
                        unknown_parent,
                        known,
                        dependency_operation.node(),
                        dependency_unknown,
                        unknown_dependency_parent,
                        dependency_known,
                        op = T::symbol(),
                    );

                    let combined = T::combine(
                        known.into_ptr(self.tape_len),
                        dependency_known.into_ptr(self.tape_len),
                    );
                    tracing::debug!(
                        "evaluated associative {} {:?}: (({:?}->{:?} {op} {}) {op} {}) to ({:?}->{:?} {op} {})",
                        T::name(),
                        operation.node(),
                        unknown,
                        unknown_parent,
                        known,
                        dependency_known,
                        unknown,
                        unknown_parent,
                        combined,
                        op = T::symbol(),
                    );

                    let int = graph.int(combined);
                    self.constants.add(int.value(), combined);

                    graph.remove_input_edges(operation.lhs());
                    graph.remove_input_edges(operation.rhs());

                    let unknown_source = graph.input_source(dependency_unknown);
                    graph.add_value_edge(unknown_source, operation.lhs());
                    graph.add_value_edge(int.value(), operation.rhs());

                    return true;
                }
            }
        }

        false
    }
}

impl Pass for AssociativeOps {
    fn pass_name(&self) -> &str {
        "associative-operations"
    }

    fn did_change(&self) -> bool {
        self.changes.did_change()
    }

    fn reset(&mut self) {
        self.constants.clear();
        self.to_be_removed.clear();
        self.changes.reset();
    }

    fn report(&self) -> ChangeReport {
        self.changes.as_report()
    }

    fn post_visit_graph(&mut self, graph: &mut Rvsdg, _visited: &HashSet<NodeId>) {
        graph.bulk_remove_nodes(&self.to_be_removed);
    }

    // `x + (y + z) ≡ (x + y) + z`
    fn visit_add(&mut self, graph: &mut Rvsdg, add: Add) {
        if self.fold_associative_operation(graph, add) {
            self.changes.inc::<"add">()
        }
    }

    // `x × (y × z) ≡ (x × y) × z`
    fn visit_mul(&mut self, graph: &mut Rvsdg, mul: Mul) {
        if self.fold_associative_operation(graph, mul) {
            self.changes.inc::<"mul">()
        }
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: Ptr) {
        self.constants.add(int.value(), value);
    }

    fn visit_byte(&mut self, _graph: &mut Rvsdg, byte: Byte, value: Cell) {
        self.constants.add(byte.value(), value);
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;
        let (mut true_visitor, mut false_visitor) =
            (Self::new(self.tape_len), Self::new(self.tape_len));

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[true_param, false_param]) in gamma.inputs().iter().zip(gamma.input_params())
        {
            let (_, source, _) = graph.get_input(input);

            if let Some(constant) = self.constants.get(source) {
                let true_param = gamma.true_branch().to_node::<InputParam>(true_param);
                true_visitor.constants.add(true_param.output(), constant);

                let false_param = gamma.false_branch().to_node::<InputParam>(false_param);
                false_visitor.constants.add(false_param.output(), constant);
            }
        }

        changed |= true_visitor.visit_graph(gamma.true_mut());
        self.changes.combine(&true_visitor.changes);
        changed |= false_visitor.visit_graph(gamma.false_mut());
        self.changes.combine(&false_visitor.changes);

        if changed {
            graph.replace_node(gamma.node(), gamma);
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut changed = false;
        let mut visitor = Self::new(self.tape_len);

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (input, param) in theta.invariant_input_pairs() {
            if let Some(constant) = self.constants.get(graph.input_source(input)) {
                visitor.constants.add(param.output(), constant);
            }
        }

        changed |= visitor.visit_graph(theta.body_mut());
        self.changes.combine(&visitor.changes);

        if changed {
            graph.replace_node(theta.node(), theta);
        }
    }
}

// TODO: Test once there's structural graph equality
