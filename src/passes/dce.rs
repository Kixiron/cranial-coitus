use crate::{
    graph::{Node, NodeExt, NodeId, Rvsdg},
    passes::{utils::ChangeReport, Pass},
    utils::HashSet,
};
use std::collections::VecDeque;

/// Removes dead code from the graph
pub struct Dce {
    changed: bool,
    nodes_removed: usize,
}

impl Dce {
    pub fn new() -> Self {
        Self {
            changed: false,
            nodes_removed: 0,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    // TODO: For subgraph'd nodes we should keep track of the ports which are actually used,
    //       since that will let us remove unused parameters from the inner subgraphs
    fn mark_nodes(
        &mut self,
        graph: &mut Rvsdg,
        stack: &mut Vec<NodeId>,
        visited: &mut HashSet<NodeId>,
        stack_len: usize,
    ) {
        stack.reserve(graph.node_len());
        visited.reserve(graph.node_len());

        while stack.len() > stack_len {
            if let Some(node_id) = stack.pop() {
                if visited.insert(node_id) {
                    stack.extend(graph.all_node_input_source_ids(node_id));

                    let stack_len = stack.len();
                    match graph.get_node_mut(node_id) {
                        Node::Gamma(gamma) => {
                            let (true_nodes, false_nodes) = (
                                gamma.true_branch().node_len(),
                                gamma.false_branch().node_len(),
                            );

                            stack.reserve(
                                1 + gamma.output_params().len() + gamma.input_params().len(),
                            );

                            // Clean up the true branch
                            {
                                // Push the gamma's true branch end node to the stack
                                stack.push(gamma.ends()[0]);
                                stack.extend(gamma.output_params().iter().map(|&[param, _]| param));

                                // Mark the nodes within the gamma's true branch
                                self.mark_nodes(gamma.true_mut(), stack, visited, stack_len);
                            }

                            // The stack shouldn't have any extra items on it
                            debug_assert_eq!(stack.len(), stack_len);

                            // Clean up the false branch
                            {
                                // Push the gamma's false branch end node to the stack
                                stack.push(gamma.ends()[1]);
                                stack.extend(gamma.output_params().iter().map(|&[_, param]| param));

                                // Mark the nodes within the gamma's false branch
                                self.mark_nodes(gamma.false_mut(), stack, visited, stack_len);
                            }

                            // TODO: Buffer
                            let inputs: Vec<_> = gamma
                                .inputs()
                                .iter()
                                .zip(gamma.input_params())
                                .map(|(&input, &params)| (input, params))
                                .enumerate()
                                .collect();

                            // Remove unused invariant params
                            let mut offset = 0;
                            for (idx, (port, [true_param, false_param])) in inputs {
                                // If neither params were used, remove them from the gamma
                                // FIXME: Once we can distinguish between params on either side of the gamma
                                //        we can remove them individually
                                if !visited.contains(&true_param) && !visited.contains(&false_param)
                                {
                                    tracing::trace!(
                                        ?true_param,
                                        ?false_param,
                                        "removed dead input {:?} from gamma {:?}",
                                        port,
                                        gamma.node(),
                                    );

                                    let index = idx - offset;
                                    gamma.inputs_mut().remove(index);
                                    gamma.input_params_mut().remove(index);

                                    gamma.true_mut().remove_node(true_param);
                                    gamma.false_mut().remove_node(false_param);

                                    offset += 1;

                                    self.changed();
                                } else {
                                    // Make sure we don't remove redundant inputs when one of them is still used
                                    // FIXME: Distinguish between gamma sides
                                    visited.insert(true_param);
                                    visited.insert(false_param);
                                }
                            }

                            if gamma.true_mut().bulk_retain_nodes(visited) {
                                self.changed();
                            }
                            self.nodes_removed += true_nodes - gamma.true_branch().node_len();

                            if gamma.false_mut().bulk_retain_nodes(visited) {
                                self.changed();
                            }
                            self.nodes_removed += false_nodes - gamma.false_branch().node_len();
                        }

                        Node::Theta(theta) => {
                            let body_nodes = theta.body().node_len();

                            stack.reserve(2 + theta.outputs_len() + theta.inputs_len());
                            visited.reserve(2 + theta.outputs_len() + theta.inputs_len());

                            // Push the theta's end node to the stack
                            stack.push(theta.end_node_id());
                            stack.push(theta.condition_id());
                            stack.extend(theta.output_param_ids());
                            stack.extend(theta.variant_input_param_ids());

                            // Mark the nodes within the theta's body
                            self.mark_nodes(theta.body_mut(), stack, visited, stack_len);

                            // TODO: Buffer
                            let invariant_inputs: Vec<_> =
                                theta.invariant_input_pair_ids().collect();

                            // Remove unused invariant params
                            for (port, param) in invariant_inputs {
                                // If the input param wasn't used, remove it from the theta
                                if !visited.contains(&param) {
                                    tracing::trace!(
                                        ?param,
                                        "removed dead invariant input {:?} from theta {:?}",
                                        port,
                                        theta.node(),
                                    );

                                    theta.remove_invariant_input(port);
                                    self.changed();
                                }
                            }

                            // Remove all dead nodes from the subgraph
                            if theta.body_mut().bulk_retain_nodes(visited) {
                                self.changed();
                            }

                            self.nodes_removed += body_nodes - theta.body().node_len();
                        }

                        _ => {}
                    }
                }
            } else {
                unreachable!()
            }
        }

        // The stack shouldn't have any extra items on it
        debug_assert_eq!(stack.len(), stack_len);
    }
}

// FIXME: Use a post-order graph walk for dce
// TODO: kill all code directly after an infinite loop
// TODO: Remove the gamma node if it's unused?
// TODO: Remove unused ports & effects from gamma nodes
// TODO: Collapse identical gamma branches?
// TODO: Remove the theta node if it's unused
// TODO: Remove effect inputs and outputs from the theta node if they're unused
// TODO: Remove unused invariant inputs
// TODO: Remove unused variant inputs and their associated output
// TODO: Hoist invariant values out of the loop and make them invariant inputs
// TODO: Demote variant inputs that don't vary into invariant inputs
// TODO: Deduplicate both variant and invariant inputs & outputs
impl Pass for Dce {
    fn pass_name(&self) -> &'static str {
        "dead-code-elimination"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.changed = false;
    }

    fn report(&self) -> ChangeReport {
        map! {
            "dead nodes" => self.nodes_removed,
        }
    }

    fn visit_graph_inner(
        &mut self,
        graph: &mut Rvsdg,
        _queue: &mut VecDeque<NodeId>,
        visited: &mut HashSet<NodeId>,
        stack: &mut Vec<NodeId>,
    ) -> bool {
        let start_nodes = graph.node_len();

        // Initialize the buffer
        stack.reserve(start_nodes / 4);
        stack.extend(graph.end_nodes());

        // Semi-recursively mark nodes
        self.mark_nodes(graph, stack, visited, 0);

        // Remove all dead nodes from the top-level graph
        if graph.bulk_retain_nodes(visited) {
            self.changed();
        }

        self.nodes_removed += start_nodes - graph.node_len();
        self.did_change()
    }
}

impl Default for Dce {
    fn default() -> Self {
        Self::new()
    }
}
