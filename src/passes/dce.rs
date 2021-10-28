use crate::{
    graph::{Node, NodeId, Rvsdg},
    passes::Pass,
};
use std::collections::{BTreeSet, VecDeque};

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

    fn mark_nodes(
        &mut self,
        graph: &mut Rvsdg,
        stack: &mut Vec<NodeId>,
        visited: &mut BTreeSet<NodeId>,
        stack_len: usize,
    ) {
        while stack.len() > stack_len {
            if let Some(node_id) = stack.pop() {
                if visited.insert(node_id) {
                    stack.extend(graph.all_node_input_source_ids(node_id));

                    let (stack_len, mut visited_len) = (stack.len(), visited.len());
                    match graph.get_node_mut(node_id) {
                        Node::Gamma(gamma) => {
                            // Clean up the true branch
                            {
                                // Push the gamma's true branch end node to the stack
                                stack.push(gamma.ends()[0]);
                                stack.extend(gamma.output_params().iter().map(|&[param, _]| param));
                                stack.extend(gamma.input_params().iter().map(|&[param, _]| param));

                                // Mark the nodes within the gamma's true branch
                                self.mark_nodes(gamma.true_mut(), stack, visited, stack_len);

                                // Only sweep though the true branch's nodes if we've actually found dead nodes within it
                                if visited.len() != visited_len {
                                    visited_len = visited.len();

                                    if gamma.true_mut().bulk_retain_nodes(visited) {
                                        self.changed();
                                    }
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

                                // Only sweep though the false branch's nodes if we've actually found dead nodes within it
                                if visited.len() != visited_len
                                    && gamma.false_mut().bulk_retain_nodes(visited)
                                {
                                    self.changed();
                                }
                            }
                        }

                        Node::Theta(theta) => {
                            // Push the theta's end node to the stack
                            stack.push(theta.end_node_id());
                            stack.extend(theta.output_param_ids());
                            stack.extend(theta.input_param_ids());

                            // Mark the nodes within the theta's body
                            self.mark_nodes(theta.body_mut(), stack, visited, stack_len);

                            // Only sweep though the body's nodes if we've actually found dead nodes within it
                            if visited.len() != visited_len
                                && theta.body_mut().bulk_retain_nodes(visited)
                            {
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
        _queue: &mut VecDeque<NodeId>,
        visited: &mut BTreeSet<NodeId>,
        stack: &mut Vec<NodeId>,
    ) -> bool {
        // debug_assert_eq!(graph.start_nodes().len(), 1);
        // debug_assert_eq!(graph.end_nodes().len(), 1);

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
