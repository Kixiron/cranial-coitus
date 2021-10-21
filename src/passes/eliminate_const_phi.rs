use crate::{
    graph::{Bool, EdgeKind, InputParam, OutputParam, OutputPort, Phi, Rvsdg, Theta},
    passes::Pass,
};
use std::collections::{BTreeMap, HashMap};

/// Evaluates constant operations within the program
pub struct ElimConstPhi {
    values: BTreeMap<OutputPort, bool>,
    changed: bool,
}

impl ElimConstPhi {
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

impl Pass for ElimConstPhi {
    fn pass_name(&self) -> &str {
        "eliminate-const-phi"
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

    fn visit_phi(&mut self, graph: &mut Rvsdg, mut phi: Phi) {
        let (mut truthy_visitor, mut falsy_visitor) = (Self::new(), Self::new());

        // For each input into the phi region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[true_param, false_param]) in phi.inputs().iter().zip(phi.input_params()) {
            let (_, source, _) = graph.get_input(input);

            if let Some(constant) = self.values.get(&source).cloned() {
                let true_param = phi.truthy().to_node::<InputParam>(true_param);
                let replaced = truthy_visitor.values.insert(true_param.value(), constant);
                debug_assert!(replaced.is_none());

                let false_param = phi.truthy().to_node::<InputParam>(false_param);
                let replaced = falsy_visitor.values.insert(false_param.value(), constant);
                debug_assert!(replaced.is_none());
            }
        }

        // If a constant condition is found, inline the chosen branch
        // Note that we don't actually visit the inlined subgraph in this iteration,
        // we leave that for successive passes to take care of
        if let Some(&condition) = self.values.get(&graph.get_input(phi.condition()).1) {
            tracing::debug!(
                "eliminated phi with constant conditional, inlining the {} branch of {:?}",
                condition,
                phi.node(),
            );

            // Choose which branch to inline into the outside graph
            let chosen_branch = if condition { phi.truthy() } else { phi.falsy() };

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

            for (&input, &params) in phi.inputs().iter().zip(phi.input_params()) {
                let param = if condition { params[0] } else { params[1] };
                let inner_output = chosen_branch.outputs(param).next().unwrap().0;
                let inlined_output = graph.input_source(input);

                output_map.insert(inner_output, inlined_output);
            }

            for (&output, &params) in phi.outputs().iter().zip(phi.output_params()) {
                let param = if condition { params[0] } else { params[1] };
                let inner_input = chosen_branch.inputs(param).next().unwrap().0;

                if let Some(inlined_input) = graph.output_dest(output) {
                    input_map.insert(inner_input, inlined_input);
                } else {
                    tracing::warn!("missing output for {:?}", output);
                }
            }

            // Inline the graph nodes, create the inlined ports and build the input/output maps
            for (node_id, mut node) in nodes {
                // Replace start nodes with the phi's input effect
                if node.is_start() {
                    let starts = phi.starts();
                    let start_node = if condition { starts[0] } else { starts[1] };

                    let branch_effect = chosen_branch.outputs(start_node).next().unwrap().0;
                    let output_effect = graph.input_source(phi.effect_in());

                    output_map.insert(branch_effect, output_effect);

                // Replace end nodes with the phi's output effect
                } else if node.is_end() {
                    let ends = phi.ends();
                    let end_node = if condition { ends[0] } else { ends[1] };

                    let branch_effect = chosen_branch.inputs(end_node).next().unwrap().0;
                    if let Some(input_effect) = graph.output_dest(phi.effect_out()) {
                        input_map.insert(branch_effect, input_effect);
                    } else {
                        tracing::warn!("missing output for {:?}", phi.effect_out());
                    }

                // Ignore input and output ports
                // Otherwise just create the required ports & inline the node
                } else {
                    for input in node.inputs_mut() {
                        let inlined = graph.input_port(node_id, EdgeKind::Value);

                        let displaced = input_map.insert(*input, inlined);
                        debug_assert!(displaced.is_none() || displaced == Some(*input));

                        *input = inlined;
                    }

                    for output in node.outputs_mut() {
                        let inlined = graph.output_port(node_id, EdgeKind::Value);

                        let displaced = output_map.insert(*output, inlined);
                        debug_assert!(displaced.is_none() || displaced == Some(*output));

                        *output = inlined;
                    }

                    graph.add_node(node_id, node);
                }
            }

            for node_id in chosen_branch.nodes() {
                for (branch_input, _, branch_output, kind) in chosen_branch.inputs(node_id) {
                    let ports = (
                        input_map.get(&branch_input).copied(),
                        output_map.get(&branch_output).copied(),
                    );

                    if let (Some(input), Some(output)) = ports {
                        graph.add_edge(output, input, kind);
                    } else {
                        tracing::error!(
                            "failed to add edge while inlining phi branch {:?}->{:?} = {:?}",
                            branch_input,
                            branch_output,
                            ports,
                        );
                    }
                }

                for (branch_output, data) in chosen_branch.outputs(node_id) {
                    if let Some((_, branch_input, kind)) = data {
                        let ports = (
                            input_map.get(&branch_input).copied(),
                            output_map.get(&branch_output).copied(),
                        );

                        if let (Some(input), Some(output)) = ports {
                            graph.add_edge(output, input, kind);
                        } else {
                            tracing::error!(
                                "failed to add edge while inlining phi branch {:?}->{:?} = {:?}",
                                branch_input,
                                branch_output,
                                ports,
                            );
                        }
                    }
                }
            }

            graph.remove_node(phi.node());
            self.changed();

        // If we can't find a constant condition for this phi, just visit all of its branches
        } else {
            truthy_visitor.visit_graph(phi.truthy_mut());
            falsy_visitor.visit_graph(phi.falsy_mut());

            self.changed |= truthy_visitor.did_change();
            self.changed |= falsy_visitor.did_change();

            graph.replace_node(phi.node(), phi);
        }

        // TODO: Propagate constants out of phi bodies?
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
                let inner_output = theta.body().outputs(param).next().unwrap().0;
                let inlined_output = graph.input_source(input);

                output_map.insert(inner_output, inlined_output);
            }

            for (&output, &param) in theta.outputs().iter().zip(theta.output_params()) {
                let inner_input = theta.body().inputs(param).next().unwrap().0;

                if let Some(inlined_input) = graph.output_dest(output) {
                    input_map.insert(inner_input, inlined_input);
                } else {
                    tracing::warn!("missing output for {:?}", output);
                }
            }

            // Inline the graph nodes, create the inlined ports and build the input/output maps
            for (node_id, mut node) in nodes {
                // Replace start nodes with the theta's input effect
                if node.is_start() {
                    let branch_effect = theta.body().outputs(theta.start()).next().unwrap().0;
                    let output_effect = graph.input_source(theta.effect_in());

                    output_map.insert(branch_effect, output_effect);

                // Replace end nodes with the theta's output effect
                } else if node.is_end() {
                    let branch_effect = theta.body().inputs(theta.end()).next().unwrap().0;
                    if let Some(input_effect) = graph.output_dest(theta.effect_out()) {
                        input_map.insert(branch_effect, input_effect);
                    } else {
                        tracing::warn!("missing output for {:?}", theta.effect_out());
                    }

                // Ignore input and output ports
                // Otherwise just create the required ports & inline the node
                } else {
                    for input in node.inputs_mut() {
                        let inlined = graph.input_port(node_id, EdgeKind::Value);

                        let displaced = input_map.insert(*input, inlined);
                        debug_assert!(displaced.is_none() || displaced == Some(*input));

                        *input = inlined;
                    }

                    for output in node.outputs_mut() {
                        let inlined = graph.output_port(node_id, EdgeKind::Value);

                        let displaced = output_map.insert(*output, inlined);
                        debug_assert!(displaced.is_none() || displaced == Some(*output));

                        *output = inlined;
                    }

                    graph.add_node(node_id, node);
                }
            }

            for node_id in theta.body().nodes() {
                for (input, _, output, kind) in theta.body().inputs(node_id) {
                    let ports = (
                        input_map.get(&input).copied(),
                        output_map.get(&output).copied(),
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
                            input_map.get(&input).copied(),
                            output_map.get(&output).copied(),
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

impl Default for ElimConstPhi {
    fn default() -> Self {
        Self::new()
    }
}
