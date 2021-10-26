use crate::{
    graph::{Add, Gamma, InputPort, Int, NodeExt, NodeId, OutputPort, Rvsdg, Theta},
    ir::Const,
    passes::Pass,
};
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// Fuses chained additions based on the law of associative addition
// TODO: Equality is also associative but it's unclear whether or not
//       that situation can actually arise within brainfuck programs
pub struct AssociativeAdd {
    values: BTreeMap<OutputPort, Const>,
    to_be_removed: HashSet<NodeId>,
    changed: bool,
}

impl AssociativeAdd {
    pub fn new() -> Self {
        Self {
            values: BTreeMap::new(),
            to_be_removed: HashSet::new(),
            changed: false,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    fn operand(&self, graph: &Rvsdg, input: InputPort) -> (InputPort, Option<i32>) {
        let (operand, output, _) = graph.get_input(input);
        let value = operand
            .as_int()
            .map(|(_, value)| value)
            .or_else(|| self.values.get(&output).and_then(Const::convert_to_i32));

        (input, value)
    }
}

impl Pass for AssociativeAdd {
    fn pass_name(&self) -> &str {
        "associative-add"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.values.clear();
        self.to_be_removed.clear();
        self.changed = false;
    }

    fn post_visit_graph(&mut self, graph: &mut Rvsdg, _visited: &BTreeSet<NodeId>) {
        graph.bulk_remove_nodes(&self.to_be_removed);
    }

    // TODO: Arithmetic folding
    // ```
    // _563 := add _562, _560
    // _564 := add _563, _560
    // _565 := add _564, _560
    // _566 := add _565, _560
    // _567 := add _566, _560
    // _568 := add _567, _560
    // _569 := add _568, _560
    // _570 := add _569, _560
    // ```
    fn visit_add(&mut self, graph: &mut Rvsdg, add: Add) {
        debug_assert_eq!(graph.incoming_count(add.node()), 2);

        let inputs = [
            self.operand(graph, add.lhs()),
            self.operand(graph, add.rhs()),
        ];

        // If one operand is constant and one is an un-evaluated expression
        if let [(_, Some(known)), (unknown, None)] | [(unknown, None), (_, Some(known))] = inputs {
            tracing::debug!("found add ({:?}: {:?} + {:?})", add.node(), unknown, known);
            let (input_node, ..) = graph.get_input(unknown);

            // If the un-evaluated expression is an add node
            if let Some(dependency_add) = input_node.as_add() {
                // Get the inputs of the dependency add
                let dependency_inputs = [
                    self.operand(graph, dependency_add.lhs()),
                    self.operand(graph, dependency_add.rhs()),
                ];

                // If one of the dependency add's inputs are unevaluated and one is constant
                if let [(_, Some(dependency_known)), (dependency_unknown, None)]
                | [(dependency_unknown, None), (_, Some(dependency_known))] = dependency_inputs
                {
                    tracing::debug!(
                        "found dependency of add ({:?}: {:?}->{:?} + {:?}), fusing it with ({:?}: {:?}->{:?} + {:?})",
                        add.node(),
                        unknown,
                        graph.port_parent(graph.input_source(unknown)),
                        known,
                        dependency_add.node(),
                        dependency_unknown,
                        graph.port_parent(graph.input_source(dependency_unknown)),
                        dependency_known,
                    );

                    let sum = known + dependency_known;
                    tracing::debug!(
                        "evaluated associative add {:?}: (({:?}->{:?} + {}) + {}) to ({:?}->{:?} + {})",
                        add.node(),
                        unknown,
                        graph.port_parent(graph.input_source(unknown)),
                        known,
                        dependency_known,
                        unknown,
                        graph.port_parent(graph.input_source(unknown)),
                        sum,
                    );

                    let int = graph.int(sum);
                    self.values.insert(int.value(), sum.into());

                    // Make a value edge between the newly fused operand and the dependency
                    // add node, removing the previous constant input from the dependency
                    let dependency_add = graph.get_node(dependency_add.node()).to_add();
                    if dependency_add.lhs() == dependency_unknown {
                        graph.remove_input_edges(dependency_add.rhs());
                        graph.add_value_edge(int.value(), dependency_add.rhs());
                    } else if dependency_add.rhs() == dependency_unknown {
                        graph.remove_input_edges(dependency_add.lhs());
                        graph.add_value_edge(int.value(), dependency_add.lhs());
                    } else {
                        unreachable!();
                    }

                    // Reroute dependents from the dependency add to the current one,
                    // the current add node is the one that we've given the fused
                    // operands to
                    graph.rewire_dependents(add.value(), dependency_add.value());
                    // Remove the now-replaced dependency add
                    // graph.remove_node(add.node());
                    self.to_be_removed.insert(add.node());

                    self.changed();
                }
            }
        }
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: i32) {
        let replaced = self.values.insert(int.value(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Int(value)));
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let (mut truthy_visitor, mut falsy_visitor) = (Self::new(), Self::new());

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[true_param, false_param]) in gamma.inputs().iter().zip(gamma.input_params())
        {
            let (_, source, _) = graph.get_input(input);

            if let Some(constant) = self.values.get(&source).cloned() {
                let true_param = gamma.true_branch().get_node(true_param).to_input_param();
                let replaced = truthy_visitor
                    .values
                    .insert(true_param.output(), constant.clone());
                debug_assert!(replaced.is_none());

                let false_param = gamma.false_branch().get_node(false_param).to_input_param();
                let replaced = falsy_visitor.values.insert(false_param.output(), constant);
                debug_assert!(replaced.is_none());
            }
        }

        // TODO: Eliminate gamma branches based on gamma condition

        truthy_visitor.visit_graph(gamma.truthy_mut());
        falsy_visitor.visit_graph(gamma.falsy_mut());
        self.changed |= truthy_visitor.did_change();
        self.changed |= falsy_visitor.did_change();

        // TODO: Propagate constants out of gamma bodies?

        graph.replace_node(gamma.node(), gamma);
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut visitor = Self::new();

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (input, param) in theta.input_pairs() {
            if let Some(constant) = self.values.get(&graph.input_source(input)).cloned() {
                let replaced = visitor.values.insert(param.output(), constant);
                debug_assert!(replaced.is_none());
            }
        }

        visitor.visit_graph(theta.body_mut());
        self.changed |= visitor.did_change();

        // TODO: Propagate constants out of theta bodies?

        graph.replace_node(theta.node(), theta);
    }
}

impl Default for AssociativeAdd {
    fn default() -> Self {
        Self::new()
    }
}
