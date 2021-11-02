use crate::{
    graph::{Node, NodeExt, NodeId, Rvsdg},
    passes::Pass,
    utils::HashSet,
};
use std::{collections::VecDeque, mem};

/// Removes dead code from the graph
pub struct Dce {
    changed: bool,
}

impl Dce {
    pub fn new() -> Self {
        Self { changed: false }
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
                            stack.reserve(
                                1 + gamma.output_params().len() + gamma.input_params().len(),
                            );

                            // Clean up the true branch
                            {
                                // Push the gamma's true branch end node to the stack
                                stack.push(gamma.ends()[0]);
                                stack.extend(gamma.output_params().iter().map(|&[param, _]| param));
                                stack.extend(gamma.input_params().iter().map(|&[param, _]| param));

                                // Mark the nodes within the gamma's true branch
                                self.mark_nodes(gamma.true_mut(), stack, visited, stack_len);

                                if gamma.true_mut().bulk_retain_nodes(visited) {
                                    self.changed();
                                }
                            }

                            // The stack shouldn't have any extra items on it
                            debug_assert_eq!(stack.len(), stack_len);

                            // Clean up the false branch
                            {
                                // Push the gamma's false branch end node to the stack
                                stack.push(gamma.ends()[1]);
                                stack.extend(gamma.output_params().iter().map(|&[_, param]| param));
                                stack.extend(gamma.input_params().iter().map(|&[param, _]| param));

                                // Mark the nodes within the gamma's false branch
                                self.mark_nodes(gamma.false_mut(), stack, visited, stack_len);

                                if gamma.false_mut().bulk_retain_nodes(visited) {
                                    self.changed();
                                }
                            }
                        }

                        Node::Theta(theta) => {
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
    fn pass_name(&self) -> &str {
        "dead-code-elimination"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.changed = false;
    }

    fn visit_graph_inner(
        &mut self,
        graph: &mut Rvsdg,
        queue: &mut VecDeque<NodeId>,
        visited: &mut HashSet<NodeId>,
        stack: &mut Vec<NodeId>,
    ) -> bool {
        mem::take(queue);

        // Initialize the buffer
        stack.extend(graph.end_nodes());

        // Semi-recursively mark nodes
        self.mark_nodes(graph, stack, visited, 0);

        // Remove all dead nodes from the top-level graph
        if graph.bulk_retain_nodes(visited) {
            self.changed();
        }

        self.did_change()
    }
}

impl Default for Dce {
    fn default() -> Self {
        Self::new()
    }
}
