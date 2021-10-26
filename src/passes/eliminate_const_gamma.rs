use crate::{
    graph::{
        Bool, EdgeKind, End, Gamma, InputParam, InputPort, Node, NodeExt, NodeId, OutputParam,
        OutputPort, Rvsdg, Start, Theta,
    },
    passes::Pass,
    utils::AssertNone,
};
use std::collections::{BTreeMap, HashMap};

/// Evaluates constant operations within the program
pub struct ElimConstGamma {
    values: BTreeMap<OutputPort, bool>,
    nodes: Vec<(NodeId, Node)>,
    input_lookup: HashMap<InputPort, Vec<InputPort>>,
    output_lookup: HashMap<OutputPort, OutputPort>,
    changed: bool,
}

impl ElimConstGamma {
    pub fn new() -> Self {
        Self {
            values: BTreeMap::new(),
            nodes: Vec::new(),
            input_lookup: HashMap::new(),
            output_lookup: HashMap::new(),
            changed: false,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }
}

impl Pass for ElimConstGamma {
    fn pass_name(&self) -> &str {
        "eliminate-const-gamma"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.values.clear();
        self.nodes.clear();
        self.input_lookup.clear();
        self.output_lookup.clear();
        self.changed = false;
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        let replaced = self.values.insert(bool.value(), value);
        debug_assert!(replaced.is_none());
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let (mut truthy_visitor, mut falsy_visitor) = (Self::new(), Self::new());

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[true_param, false_param]) in gamma.inputs().iter().zip(gamma.input_params())
        {
            let (_, source, _) = graph.get_input(input);

            if let Some(constant) = self.values.get(&source).cloned() {
                let true_param = gamma.true_branch().to_node::<InputParam>(true_param);
                let replaced = truthy_visitor.values.insert(true_param.output(), constant);
                debug_assert!(replaced.is_none());

                let false_param = gamma.true_branch().to_node::<InputParam>(false_param);
                let replaced = falsy_visitor.values.insert(false_param.output(), constant);
                debug_assert!(replaced.is_none());
            }
        }

        // If a constant condition is found, inline the chosen branch
        // Note that we don't actually visit the inlined subgraph in this iteration,
        // we leave that for successive passes to take care of
        if let Some(&condition) = self.values.get(&graph.get_input(gamma.condition()).1) {
            tracing::debug!(
                "eliminated gamma with constant conditional, inlining the {} branch of {:?}",
                condition,
                gamma.node(),
            );

            // Choose which branch to inline into the outside graph
            let chosen_branch = if condition {
                gamma.true_branch()
            } else {
                gamma.false_branch()
            };

            debug_assert!(
                self.nodes.is_empty()
                    && self.input_lookup.is_empty()
                    && self.output_lookup.is_empty()
            );
            self.nodes.extend(
                chosen_branch
                    .iter_nodes()
                    .map(|(node_id, node)| (node_id, node.clone())),
            );

            for (&input, &params) in gamma.inputs().iter().zip(gamma.input_params()) {
                let param = if condition { params[0] } else { params[1] };
                let input_param = chosen_branch.to_node::<InputParam>(param);
                let inlined_output = graph.input_source(input);

                self.output_lookup
                    .insert(input_param.output(), inlined_output)
                    .debug_unwrap_none();
            }

            for (&output, &params) in gamma.outputs().iter().zip(gamma.output_params()) {
                let param = if condition { params[0] } else { params[1] };
                let output_param = chosen_branch.to_node::<OutputParam>(param);

                self.input_lookup
                    .insert(
                        output_param.input(),
                        graph.output_dest(output).collect::<Vec<_>>(),
                    )
                    .debug_unwrap_none();
            }

            // Inline the graph nodes, create the inlined ports and build the input/output maps
            for (node_id, mut node) in self.nodes.drain(..) {
                // Replace start nodes with the gamma's input effect
                if node.is_start() {
                    let starts = gamma.starts();
                    let start_id = if condition { starts[0] } else { starts[1] };
                    let start = chosen_branch.to_node::<Start>(start_id);

                    let output_effect = graph.input_source(gamma.effect_in());
                    self.output_lookup
                        .insert(start.effect(), output_effect)
                        .debug_unwrap_none();

                // Replace end nodes with the gamma's output effect
                } else if node.is_end() {
                    let ends = gamma.ends();
                    let end_id = if condition { ends[0] } else { ends[1] };
                    let end = chosen_branch.to_node::<End>(end_id);

                    self.input_lookup
                        .insert(
                            end.effect_in(),
                            graph.output_dest(gamma.effect_out()).collect(),
                        )
                        .debug_unwrap_none();

                // Ignore input and output ports
                // Otherwise just create the required ports & inline the node
                } else if !node.is_input_port() && !node.is_output_port() {
                    for input in node.inputs_mut() {
                        let inlined = graph.input_port(node_id, EdgeKind::Value);

                        let replaced = self.input_lookup.insert(*input, vec![inlined]);
                        debug_assert!(
                            replaced.is_none(),
                            "replaced value {:?} for input port {:?} with {:?}",
                            replaced,
                            input,
                            inlined,
                        );

                        *input = inlined;
                    }

                    for output in node.outputs_mut() {
                        let inlined = graph.output_port(node_id, EdgeKind::Value);

                        let replaced = self.output_lookup.insert(*output, inlined);
                        debug_assert!(
                            replaced.is_none() ,
                            "replaced value {:?} for output port {:?} with {:?}\ninputs: {:?}\noutputs: {:?}",
                            replaced,
                            output,
                            inlined,
                            self.input_lookup,
                            self.output_lookup,
                        );

                        *output = inlined;
                    }

                    graph.add_node(node_id, node);
                }
            }

            for node_id in chosen_branch.node_ids() {
                for (branch_input, _, branch_output, kind) in chosen_branch.inputs(node_id) {
                    let (inputs, output) = (
                        self.input_lookup.get(&branch_input).into_iter().flatten(),
                        self.output_lookup.get(&branch_output).copied(),
                    );

                    if let Some(output) = output {
                        for &input in inputs {
                            graph.add_edge(output, input, kind);
                        }
                    } else {
                        tracing::error!(
                            "failed to add edge while inlining gamma branch, missing output port for {:?} (inputs: {:?})",
                            branch_output,
                            inputs.collect::<Vec<_>>(),
                        );
                    }
                }
            }

            self.nodes.clear();
            self.input_lookup.clear();
            self.output_lookup.clear();

            graph.remove_node(gamma.node());
            self.changed();

        // If we can't find a constant condition for this gamma, just visit all of its branches
        } else {
            truthy_visitor.visit_graph(gamma.truthy_mut());
            falsy_visitor.visit_graph(gamma.falsy_mut());

            self.changed |= truthy_visitor.did_change();
            self.changed |= falsy_visitor.did_change();

            graph.replace_node(gamma.node(), gamma);
        }

        // TODO: Propagate constants out of gamma bodies?
    }

    // FIXME: Inlining gamma bodies is broken rn
    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut visitor = Self::new();

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (input, param) in theta.input_pairs() {
            if let Some(constant) = self.values.get(&graph.input_source(input)).cloned() {
                visitor
                    .values
                    .insert(param.output(), constant)
                    .debug_unwrap_none();
            }
        }

        visitor.visit_graph(theta.body_mut());
        self.changed |= visitor.did_change();

        let cond_out = theta.condition();
        let cond_source = theta.body().input_source(cond_out.input());
        let cond_value = visitor.values.get(&cond_source).copied();

        // If the theta's condition is `false`, it will never loop and we can inline the body
        if cond_value == Some(false) {
            tracing::debug!(
                "eliminated theta {:?} with false as its loop variable, inlining its body once",
                theta.node(),
            );

            // TODO: Factor subgraph inlining into a function
            debug_assert!(
                self.nodes.is_empty()
                    && self.input_lookup.is_empty()
                    && self.output_lookup.is_empty()
            );
            self.nodes.extend(
                theta
                    .body()
                    .iter_nodes()
                    .map(|(node_id, node)| (node_id, node.clone())),
            );

            for (input, param) in theta.input_pairs() {
                let inlined_output = graph.input_source(input);

                self.output_lookup
                    .insert(param.output(), inlined_output)
                    .debug_unwrap_none();
            }

            for (output, param) in theta.output_pairs() {
                self.input_lookup
                    .insert(param.input(), graph.output_dest(output).collect::<Vec<_>>())
                    .debug_unwrap_none();
            }

            // Inline the graph nodes, create the inlined ports and build the input/output maps
            for (node_id, mut node) in self.nodes.drain(..) {
                // Replace start nodes with the gamma's input effect
                if node.is_start() {
                    let start = theta.start_node();
                    let output_effect = graph.input_source(theta.input_effect().unwrap());

                    self.output_lookup
                        .insert(start.effect(), output_effect)
                        .debug_unwrap_none();

                // Replace end nodes with the gamma's output effect
                } else if node.is_end() {
                    let end = theta.end_node();
                    self.input_lookup
                        .insert(
                            end.effect_in(),
                            graph.output_dest(theta.output_effect().unwrap()).collect(),
                        )
                        .debug_unwrap_none();

                // Ignore input and output ports
                // Otherwise just create the required ports & inline the node
                } else if !node.is_input_port() && !node.is_output_port() {
                    for input in node.inputs_mut() {
                        let inlined = graph.input_port(node_id, EdgeKind::Value);

                        let replaced = self.input_lookup.insert(*input, vec![inlined]);
                        debug_assert!(
                            replaced.is_none(),
                            "replaced value {:?} for input port {:?} with {:?}",
                            replaced,
                            input,
                            inlined,
                        );

                        *input = inlined;
                    }

                    for output in node.outputs_mut() {
                        let inlined = graph.output_port(node_id, EdgeKind::Value);

                        let replaced = self.output_lookup.insert(*output, inlined);
                        debug_assert!(
                            replaced.is_none() ,
                            "replaced value {:?} for output port {:?} with {:?}\ninputs: {:?}\noutputs: {:?}",
                            replaced,
                            output,
                            inlined,
                            self.input_lookup,
                            self.output_lookup,
                        );

                        *output = inlined;
                    }

                    graph.add_node(node_id, node);
                }
            }

            for node_id in theta.body().node_ids() {
                for (branch_input, _, branch_output, kind) in theta.body().inputs(node_id) {
                    let (inputs, output) = (
                        self.input_lookup.get(&branch_input).into_iter().flatten(),
                        self.output_lookup.get(&branch_output).copied(),
                    );

                    if let Some(output) = output {
                        for &input in inputs {
                            graph.add_edge(output, input, kind);
                        }
                    } else {
                        tracing::error!(
                            "failed to add {} edge while inlining theta branch, missing output port for {:?} (inputs: {:?})",
                            kind,
                            branch_output,
                            inputs.collect::<Vec<_>>(),
                        );
                    }
                }
            }

            self.nodes.clear();
            self.input_lookup.clear();
            self.output_lookup.clear();

            graph.remove_node(theta.node());
            self.changed();
        } else {
            // TODO: Propagate constants out of theta bodies?
            graph.replace_node(theta.node(), theta);
        }
    }
}

impl Default for ElimConstGamma {
    fn default() -> Self {
        Self::new()
    }
}
