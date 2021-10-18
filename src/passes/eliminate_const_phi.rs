use crate::{
    graph::{Bool, NodeId, Phi, Rvsdg, Theta},
    passes::Pass,
};
use std::collections::HashMap;

/// Evaluates constant operations within the program
pub struct ElimConstPhi {
    values: HashMap<NodeId, bool>,
    changed: bool,
}

impl ElimConstPhi {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
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
        let replaced = self.values.insert(bool.node(), value);
        debug_assert!(replaced.is_none());
    }

    fn visit_phi(&mut self, graph: &mut Rvsdg, mut phi: Phi) {
        let (mut truthy_visitor, mut falsy_visitor) = (Self::new(), Self::new());

        // For each input into the phi region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[truthy_param, falsy_param]) in phi.inputs().iter().zip(phi.input_params()) {
            let (input_node, _, _) = graph.get_input(input);
            let input_node_id = input_node.node_id();

            if let Some(constant) = self.values.get(&input_node_id).cloned() {
                let replaced = truthy_visitor.values.insert(truthy_param, constant);
                debug_assert!(replaced.is_none());

                let replaced = falsy_visitor.values.insert(falsy_param, constant);
                debug_assert!(replaced.is_none());
            }
        }

        // If a constant condition is found, inline the chosen branch
        // Note that we don't actually visit the inlined subgraph in this iteration,
        // we leave that for successive passes to take care of
        if let Some(&condition) = self
            .values
            .get(&graph.get_input(phi.condition()).0.node_id())
        {
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
                    }

                // Replace input ports with the outer region's output ports
                } else if node.is_input_port() {
                    for (&input, &params) in phi.inputs().iter().zip(phi.input_params()) {
                        let param = if condition { params[0] } else { params[1] };
                        let inner_output = chosen_branch.outputs(param).next().unwrap().0;
                        let inlined_output = graph.input_source(input);

                        output_map.insert(inner_output, inlined_output);
                    }

                // Replace output ports with the outer region's input ports
                } else if node.is_output_port() {
                    for (&output, &params) in phi.outputs().iter().zip(phi.output_params()) {
                        let param = if condition { params[0] } else { params[1] };
                        let inner_input = chosen_branch.inputs(param).next().unwrap().0;

                        if let Some(inlined_input) = graph.output_dest(output) {
                            input_map.insert(inner_input, inlined_input);
                        }
                    }

                // Otherwise just create the required ports & inline the node
                } else {
                    for input in node.inputs_mut() {
                        let inlined = graph.input_port(node_id);

                        let displaced = input_map.insert(*input, inlined);
                        debug_assert!(displaced.is_none());

                        *input = inlined;
                    }

                    for output in node.outputs_mut() {
                        let inlined = graph.output_port(node_id);

                        let displaced = output_map.insert(*output, inlined);
                        debug_assert!(displaced.is_none());

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
            let (input_node, _, _) = graph.get_input(input);
            let input_node_id = input_node.node_id();

            if let Some(constant) = self.values.get(&input_node_id).cloned() {
                let replaced = visitor.values.insert(param, constant);
                debug_assert!(replaced.is_none());
            }
        }

        visitor.visit_graph(theta.body_mut());
        self.changed |= visitor.did_change();

        // TODO: Propagate constants out of theta bodies?

        graph.replace_node(theta.node(), theta);
    }
}

impl Default for ElimConstPhi {
    fn default() -> Self {
        Self::new()
    }
}
