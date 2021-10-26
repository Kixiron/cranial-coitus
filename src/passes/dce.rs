use crate::{
    graph::{EdgeKind, Gamma, Node, NodeExt, NodeId, Rvsdg, Theta},
    passes::Pass,
};
use std::collections::{BTreeSet, VecDeque};

/// Removes dead code from the graph
pub struct Dce {
    changed: bool,
    stack_buf: VecDeque<NodeId>,
    visited_buf: BTreeSet<NodeId>,
    buffer_buf: Vec<NodeId>,
}

impl Dce {
    pub fn new() -> Self {
        Self {
            changed: false,
            stack_buf: VecDeque::new(),
            visited_buf: BTreeSet::new(),
            buffer_buf: Vec::new(),
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }
}

// FIXME: Use a post-order graph walk for dce
// TODO: kill all code directly after an infinite loop
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

    fn post_visit_graph(&mut self, graph: &mut Rvsdg, visited: &BTreeSet<NodeId>) {
        let nodes: Vec<_> = graph.node_ids().collect();

        for node_id in nodes {
            let node = graph.get_node(node_id);

            // If the node hasn't been visited then it's dead. `Pass` operates off of a
            // dfs, so any node not visited has no incoming or outgoing edges to it.
            // As an alternative for if the dependent nodes of a given node are removed,
            // we check if there aren't incoming or outgoing edges
            if !visited.contains(&node_id)
                || (graph.incoming_count(node_id) == 0
                    && graph.outgoing_count(node_id) == 0
                    && !matches!(
                        node,
                        Node::Start(..)
                            | Node::End(..)
                            | Node::InputPort(..)
                            | Node::OutputPort(..)
                    ))
            {
                tracing::debug!(
                    visited = visited.contains(&node_id),
                    incoming = graph.incoming_count(node_id),
                    outgoing = graph.outgoing_count(node_id),
                    node = ?graph.get_node(node_id),
                    "removed dead node {:?}",
                    node_id,
                );
                graph.remove_node(node_id);
                self.changed();
            }
        }
    }

    // TODO: Remove the gamma node if it's unused?
    // TODO: Remove unused ports & effects from gamma nodes
    // TODO: Collapse identical gamma branches?
    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let (mut truthy_visitor, mut falsy_visitor) = (Self::new(), Self::new());

        self.stack_buf.extend(
            gamma
                .input_params()
                .iter()
                .chain(gamma.output_params())
                .map(|&[truthy, _]| truthy),
        );
        truthy_visitor.visit_graph_inner(
            gamma.truthy_mut(),
            &mut self.stack_buf,
            &mut self.visited_buf,
            &mut self.buffer_buf,
        );
        self.changed |= truthy_visitor.did_change();

        self.stack_buf.extend(
            gamma
                .input_params()
                .iter()
                .chain(gamma.output_params())
                .map(|&[_, falsy]| falsy),
        );
        falsy_visitor.visit_graph_inner(
            gamma.falsy_mut(),
            &mut self.stack_buf,
            &mut self.visited_buf,
            &mut self.buffer_buf,
        );
        self.changed |= falsy_visitor.did_change();

        let branch_is_empty = |graph: &Rvsdg, start| {
            let start = graph.get_node(start).to_start();

            graph
                .get_output(start.effect())
                .map_or(false, |(consumer, _, _)| consumer.is_end())
        };

        let true_is_empty = branch_is_empty(gamma.true_branch(), gamma.starts()[0]);
        let false_is_empty = branch_is_empty(gamma.false_branch(), gamma.starts()[1]);

        if true_is_empty && false_is_empty {
            tracing::debug!("removing an empty gamma {:?}", gamma.node());

            graph.splice_ports(gamma.effect_in(), gamma.effect_out());

            for (&input_port, &param) in gamma.inputs().iter().zip(gamma.input_params()) {
                let truthy_param = gamma.true_branch().get_node(param[0]).to_input_param();
                let falsy_param = gamma.false_branch().get_node(param[1]).to_input_param();

                if let Some((Node::OutputPort(output), _, EdgeKind::Value)) =
                    gamma.true_branch().get_output(truthy_param.output())
                {
                    let output_port = gamma.outputs().iter().zip(gamma.output_params()).find_map(
                        |(&output_port, &param)| (param[0] == output.node()).then(|| output_port),
                    );

                    if let Some(output_port) = output_port {
                        tracing::debug!(
                            "splicing gamma input to output passthrough {:?}->{:?}",
                            input_port,
                            output_port,
                        );

                        graph.splice_ports(input_port, output_port);
                    }
                } else if let Some((Node::OutputPort(output), _, EdgeKind::Value)) =
                    gamma.false_branch().get_output(falsy_param.output())
                {
                    let output_port = gamma.outputs().iter().zip(gamma.output_params()).find_map(
                        |(&output_port, &param)| (param[1] == output.node()).then(|| output_port),
                    );

                    if let Some(output_port) = output_port {
                        tracing::debug!(
                            "splicing gamma input to output passthrough {:?}->{:?}",
                            input_port,
                            output_port,
                        );

                        graph.splice_ports(input_port, output_port);
                    }
                }
            }

            graph.remove_node(gamma.node());
            self.changed();
        } else {
            graph.replace_node(gamma.node(), gamma);
        }
    }

    // TODO: Remove the theta node if it's unused
    // TODO: Remove effect inputs and outputs from the theta node if they're unused
    // TODO: Remove unused invariant inputs
    // TODO: Remove unused variant inputs and their associated output
    // TODO: Hoist invariant values out of the loop and make them invariant inputs
    // TODO: Demote variant inputs that don't vary into invariant inputs
    // TODO: Deduplicate both variant and invariant inputs & outputs
    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut visitor = Self::new();

        // Add all inputs & outputs to the list of nodes to be visited by the subgraph visitor
        self.stack_buf
            .extend(theta.output_param_ids().chain(theta.input_param_ids()));

        visitor.visit_graph_inner(
            theta.body_mut(),
            &mut self.stack_buf,
            &mut self.visited_buf,
            &mut self.buffer_buf,
        );
        self.changed |= visitor.did_change();

        graph.replace_node(theta.node(), theta);
    }

    // TODO: Remove unused {Input, Output}Params
}

impl Default for Dce {
    fn default() -> Self {
        Self::new()
    }
}
