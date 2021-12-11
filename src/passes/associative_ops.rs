use crate::{
    graph::{
        Add, Gamma, InputParam, InputPort, Int, Mul, Node, NodeExt, NodeId, OutputPort, Rvsdg,
        Theta,
    },
    ir::Const,
    passes::{utils::BinOp, Pass},
    utils::{AssertNone, HashMap, HashSet},
};

/// Fuses chained additions based on the law of associative addition
// TODO: Equality is also associative but it's unclear whether or not
//       that situation can actually arise within brainfuck programs
pub struct AssociativeOps {
    // TODO: ConstantStore
    values: HashMap<OutputPort, Const>,
    to_be_removed: HashSet<NodeId>,
    changed: bool,
}

impl AssociativeOps {
    pub fn new() -> Self {
        Self {
            values: HashMap::with_hasher(Default::default()),
            to_be_removed: HashSet::with_hasher(Default::default()),
            changed: false,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    fn operand(&self, graph: &Rvsdg, input: InputPort) -> (InputPort, Option<u32>) {
        let (operand, output, _) = graph.get_input(input);
        let value = operand
            .as_int()
            .map(|(_, value)| value)
            .or_else(|| self.values.get(&output).and_then(Const::convert_to_u32));

        (input, value)
    }

    fn fold_associative_operation<T>(&mut self, graph: &mut Rvsdg, operation: T)
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

                    let combined = T::combine(known, dependency_known);
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
                    self.values.insert(int.value(), combined.into());

                    graph.remove_input_edges(operation.lhs());
                    graph.remove_input_edges(operation.rhs());

                    let unknown_source = graph.input_source(dependency_unknown);
                    graph.add_value_edge(unknown_source, operation.lhs());
                    graph.add_value_edge(int.value(), operation.rhs());

                    self.changed();
                }
            }
        }
    }
}

impl Pass for AssociativeOps {
    fn pass_name(&self) -> &str {
        "associative-operations"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.values.clear();
        self.to_be_removed.clear();
        self.changed = false;
    }

    fn post_visit_graph(&mut self, graph: &mut Rvsdg, _visited: &HashSet<NodeId>) {
        graph.bulk_remove_nodes(&self.to_be_removed);
    }

    // `x + (y + z) ≡ (x + y) + z`
    fn visit_add(&mut self, graph: &mut Rvsdg, add: Add) {
        self.fold_associative_operation(graph, add);
    }

    // `x × (y × z) ≡ (x × y) × z`
    fn visit_mul(&mut self, graph: &mut Rvsdg, mul: Mul) {
        self.fold_associative_operation(graph, mul);
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: u32) {
        let replaced = self.values.insert(int.value(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Int(value)));
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;
        let (mut true_visitor, mut false_visitor) = (Self::new(), Self::new());

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[true_param, false_param]) in gamma.inputs().iter().zip(gamma.input_params())
        {
            let (_, source, _) = graph.get_input(input);

            if let Some(constant) = self.values.get(&source).copied() {
                let true_param = gamma.true_branch().to_node::<InputParam>(true_param);
                true_visitor
                    .values
                    .insert(true_param.output(), constant)
                    .debug_unwrap_none();

                let false_param = gamma.false_branch().to_node::<InputParam>(false_param);
                false_visitor
                    .values
                    .insert(false_param.output(), constant)
                    .debug_unwrap_none();
            }
        }

        changed |= true_visitor.visit_graph(gamma.true_mut());
        changed |= false_visitor.visit_graph(gamma.false_mut());

        if changed {
            graph.replace_node(gamma.node(), gamma);
            self.changed();
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut changed = false;
        let mut visitor = Self::new();

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (input, param) in theta.invariant_input_pairs() {
            if let Some(constant) = self.values.get(&graph.input_source(input)).cloned() {
                visitor
                    .values
                    .insert(param.output(), constant)
                    .debug_unwrap_none();
            }
        }

        visitor.visit_graph(theta.body_mut());
        changed |= visitor.did_change();

        if changed {
            graph.replace_node(theta.node(), theta);
            self.changed();
        }
    }
}

impl Default for AssociativeOps {
    fn default() -> Self {
        Self::new()
    }
}

// TODO: Test once there's structural graph equality
