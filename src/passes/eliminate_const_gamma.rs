use crate::{
    graph::{Bool, EdgeKind, End, Gamma, InputParam, OutputParam, OutputPort, Rvsdg, Start, Theta},
    passes::Pass,
};
use std::collections::{BTreeMap, HashMap};

/// Evaluates constant operations within the program
pub struct ElimConstGamma {
    values: BTreeMap<OutputPort, bool>,
    changed: bool,
}

impl ElimConstGamma {
    pub fn new() -> Self {
        Self {
            values: BTreeMap::new(),
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
                let replaced = truthy_visitor.values.insert(true_param.value(), constant);
                debug_assert!(replaced.is_none());

                let false_param = gamma.true_branch().to_node::<InputParam>(false_param);
                let replaced = falsy_visitor.values.insert(false_param.value(), constant);
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

            // TODO: Reuse this buffer
            let nodes: Vec<_> = chosen_branch
                .iter_nodes()
                .map(|(node_id, node)| (node_id, node.clone()))
                .collect();

            // Create maps to associate the inlined ports with the old ones
            // TODO: Reuse these buffers
            let (mut input_map, mut output_map) = (
                HashMap::with_capacity(nodes.len()),
                HashMap::with_capacity(nodes.len()),
            );

            for (&input, &params) in gamma.inputs().iter().zip(gamma.input_params()) {
                let param = if condition { params[0] } else { params[1] };
                let input_param = chosen_branch.to_node::<InputParam>(param);
                let inlined_output = graph.input_source(input);

                output_map.insert((param, input_param.value()), inlined_output);
            }

            for (&output, &params) in gamma.outputs().iter().zip(gamma.output_params()) {
                let param = if condition { params[0] } else { params[1] };
                let output_param = chosen_branch.to_node::<OutputParam>(param);

                if let Some(inlined_input) = graph.output_dest(output) {
                    input_map.insert((param, output_param.value()), inlined_input);
                } else {
                    tracing::warn!("missing output destination for output param {:?}", output);
                }
            }

            // Inline the graph nodes, create the inlined ports and build the input/output maps
            for (node_id, mut node) in nodes {
                // Replace start nodes with the gamma's input effect
                if node.is_start() {
                    let starts = gamma.starts();
                    let start_id = if condition { starts[0] } else { starts[1] };
                    let start = chosen_branch.to_node::<Start>(start_id);

                    let output_effect = graph.input_source(gamma.effect_in());
                    output_map.insert((start.node(), start.effect()), output_effect);

                // Replace end nodes with the gamma's output effect
                } else if node.is_end() {
                    let ends = gamma.ends();
                    let end_id = if condition { ends[0] } else { ends[1] };
                    let end = chosen_branch.to_node::<End>(end_id);

                    if let Some(input_effect) = graph.output_dest(gamma.effect_out()) {
                        input_map.insert((end.node(), end.effect_in()), input_effect);
                    } else {
                        tracing::warn!("missing output for {:?}", gamma.effect_out());
                    }

                // Ignore input and output ports
                // Otherwise just create the required ports & inline the node
                } else if !node.is_input_port() && !node.is_output_port() {
                    for input in node.inputs_mut() {
                        let inlined = graph.input_port(node_id, EdgeKind::Value);

                        let displaced = input_map.insert((node_id, *input), inlined);
                        debug_assert!(
                            displaced.is_none() || displaced == Some(inlined),
                            "displaced value {:?} for input port {:?} with {:?}",
                            displaced,
                            input,
                            inlined,
                        );

                        *input = inlined;
                    }

                    for output in node.outputs_mut() {
                        let inlined = graph.output_port(node_id, EdgeKind::Value);

                        let displaced = output_map.insert((node_id, *output), inlined);
                        debug_assert!(
                            displaced.is_none() || displaced == Some(inlined),
                            "displaced value {:?} for output port {:?} with {:?}\ninputs: {:?}\noutputs: {:?}",
                            displaced,
                            output,
                            inlined,
                            input_map,
                            output_map,
                        );

                        *output = inlined;
                    }

                    graph.add_node(node_id, node);
                }
            }

            for node_id in chosen_branch.node_ids() {
                for (branch_input, _, branch_output, kind) in chosen_branch.inputs(node_id) {
                    let ports = (
                        input_map.get(&(node_id, branch_input)).copied(),
                        output_map.get(&(node_id, branch_output)).copied(),
                    );

                    if let (Some(input), Some(output)) = ports {
                        graph.add_edge(output, input, kind);
                    } else {
                        tracing::error!(
                            "failed to add edge while inlining gamma branch {:?}->{:?} = {:?}",
                            branch_input,
                            branch_output,
                            ports,
                        );
                    }
                }

                for (branch_output, data) in chosen_branch.outputs(node_id) {
                    if let Some((_, branch_input, kind)) = data {
                        let ports = (
                            input_map.get(&(node_id, branch_input)).copied(),
                            output_map.get(&(node_id, branch_output)).copied(),
                        );

                        if let (Some(input), Some(output)) = ports {
                            graph.add_edge(output, input, kind);
                        } else {
                            tracing::error!(
                                "failed to add edge while inlining gamma branch {:?}->{:?} = {:?}",
                                branch_input,
                                branch_output,
                                ports,
                            );
                        }
                    }
                }
            }

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

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut visitor = Self::new();

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &param) in theta.inputs().iter().zip(theta.input_params()) {
            let (_, source, _) = graph.get_input(input);

            if let Some(constant) = self.values.get(&source).cloned() {
                let param = theta.body().to_node::<InputParam>(param);
                let replaced = visitor.values.insert(param.value(), constant);
                debug_assert!(replaced.is_none());
            }
        }

        visitor.visit_graph(theta.body_mut());
        self.changed |= visitor.did_change();

        let cond_out = theta.body().to_node::<OutputParam>(theta.condition());
        let cond_input = theta.body().get_input(cond_out.value()).1;
        let cond_value = visitor.values.get(&cond_input).copied();

        // If the theta's condition is `false`, it will never loop and we can inline the body
        if cond_value == Some(false) {
            // TODO: Factor subgraph inlining into a function
            // TODO: Reuse this buffer
            let nodes: Vec<_> = theta
                .body()
                .iter_nodes()
                .map(|(node_id, node)| (node_id, node.clone()))
                .collect();

            // Create maps to associate the inlined ports with the old ones
            // TODO: Reuse these buffers
            let (mut input_map, mut output_map) = (
                HashMap::with_capacity(nodes.len()),
                HashMap::with_capacity(nodes.len()),
            );

            for (&input, &param) in theta.inputs().iter().zip(theta.input_params()) {
                let input_param = theta.body().to_node::<InputParam>(param);
                let inlined_output = graph.input_source(input);

                output_map.insert((param, input_param.value()), inlined_output);
            }

            for (&output, &param) in theta.outputs().iter().zip(theta.output_params()) {
                let output_param = theta.body().to_node::<OutputParam>(param);

                if let Some(inlined_input) = graph.output_dest(output) {
                    input_map.insert((param, output_param.value()), inlined_input);
                } else {
                    tracing::warn!("missing output for {:?}", output);
                }
            }

            // Inline the graph nodes, create the inlined ports and build the input/output maps
            for (node_id, mut node) in nodes {
                // Replace start nodes with the theta's input effect
                if node.is_start() {
                    let start = theta.body().to_node::<Start>(theta.start());

                    let output_effect = graph.input_source(theta.effect_in());
                    output_map.insert((node_id, start.effect()), output_effect);

                // Replace end nodes with the gamma's output effect
                } else if node.is_end() {
                    let end = theta.body().to_node::<End>(theta.end());

                    if let Some(input_effect) = graph.output_dest(theta.effect_out()) {
                        input_map.insert((node_id, end.effect_in()), input_effect);
                    } else {
                        tracing::warn!("missing output effect for theta {:?}", theta.effect_out());
                    }

                // Ignore input and output ports
                // Otherwise just create the required ports & inline the node
                } else if !node.is_input_port() && !node.is_output_port() {
                    for input in node.inputs_mut() {
                        let inlined = graph.input_port(node_id, EdgeKind::Value);

                        let displaced = input_map.insert((node_id, *input), inlined);
                        debug_assert!(displaced.is_none() || displaced == Some(inlined));

                        *input = inlined;
                    }

                    for output in node.outputs_mut() {
                        let inlined = graph.output_port(node_id, EdgeKind::Value);

                        let displaced = output_map.insert((node_id, *output), inlined);
                        debug_assert!(displaced.is_none() || displaced == Some(inlined));

                        *output = inlined;
                    }

                    graph.add_node(node_id, node);
                }
            }

            for node_id in theta.body().node_ids() {
                for (input, _, output, kind) in theta.body().inputs(node_id) {
                    let ports = (
                        input_map.get(&(node_id, input)).copied(),
                        output_map.get(&(node_id, output)).copied(),
                    );

                    if let (Some(input), Some(output)) = ports {
                        graph.add_edge(output, input, kind);
                    } else {
                        tracing::error!(
                            "failed to add edge while inlining theta body {:?}->{:?} = {:?}",
                            input,
                            output,
                            ports,
                        );
                    }
                }

                for (output, data) in theta.body().outputs(node_id) {
                    if let Some((_, input, kind)) = data {
                        let ports = (
                            input_map.get(&(node_id, input)).copied(),
                            output_map.get(&(node_id, output)).copied(),
                        );

                        if let (Some(input), Some(output)) = ports {
                            graph.add_edge(output, input, kind);
                        } else {
                            tracing::error!(
                                "failed to add edge while inlining theta body {:?}->{:?} = {:?}",
                                input,
                                output,
                                ports,
                            );
                        }
                    }
                }
            }

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
