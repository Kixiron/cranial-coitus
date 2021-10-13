use crate::{
    graph::{Node, NodeId, Phi, Rvsdg, Theta},
    passes::Pass,
};
use std::collections::HashSet;

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
}

impl Pass for Dce {
    fn pass_name(&self) -> &str {
        "dead-code-elimination"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn post_visit_graph(&mut self, graph: &mut Rvsdg, visited: &HashSet<NodeId>) {
        let nodes: Vec<_> = graph.nodes().collect();

        for node_id in nodes {
            // If the node hasn't been visited then it's dead. `Pass` operates off of a
            // dfs, so any node not visited has no incoming or outgoing edges to it.
            // As an alternative for if the dependent nodes of a given node are removed,
            // we check if there aren't incoming or outgoing edges
            if !visited.contains(&node_id)
                || (graph.incoming_count(node_id) == 0
                    && graph.outgoing_count(node_id) == 0
                    && !matches!(
                        graph.get_node(node_id),
                        Node::Start(..)
                            | Node::End(..)
                            | Node::InputPort(..)
                            | Node::OutputPort(..)
                    ))
            {
                tracing::debug!("removed dead node {:?}", node_id);
                graph.remove_node(node_id);
                self.changed();
            }
        }
    }

    // TODO: Remove the phi node if it's unused
    fn visit_phi(&mut self, graph: &mut Rvsdg, mut phi: Phi) {
        let (mut truthy_visitor, mut falsy_visitor) = (Self::new(), Self::new());
        let (truthy_params, falsy_params) = phi
            .input_params()
            .iter()
            .chain(phi.output_params())
            .map(|&[truthy, falsy]| (truthy, falsy))
            .unzip();

        truthy_visitor.visit_graph_inner(phi.truthy_mut(), truthy_params);
        falsy_visitor.visit_graph_inner(phi.falsy_mut(), falsy_params);
        self.changed |= truthy_visitor.did_change();
        self.changed |= falsy_visitor.did_change();

        graph.replace_node(phi.node(), phi);
    }

    // TODO: Remove the theta node if it's unused
    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut visitor = Self::new();
        let params = theta
            .input_params()
            .iter()
            .chain(theta.output_params())
            .copied()
            .collect();

        visitor.visit_graph_inner(theta.body_mut(), params);
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
