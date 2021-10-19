use crate::{
    graph::{EdgeKind, Node, NodeId, Phi, Rvsdg, Theta},
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

    // TODO: Remove the phi node if it's unused?
    // TODO: Remove unused ports & effects from phi nodes
    // TODO: Collapse identical phi branches?
    fn visit_phi(&mut self, graph: &mut Rvsdg, mut phi: Phi) {
        let (mut truthy_visitor, mut falsy_visitor) = (Self::new(), Self::new());

        self.stack_buf.extend(
            phi.input_params()
                .iter()
                .chain(phi.output_params())
                .map(|&[truthy, _]| truthy),
        );
        truthy_visitor.visit_graph_inner(
            phi.truthy_mut(),
            &mut self.stack_buf,
            &mut self.visited_buf,
            &mut self.buffer_buf,
        );
        self.changed |= truthy_visitor.did_change();

        self.stack_buf.extend(
            phi.input_params()
                .iter()
                .chain(phi.output_params())
                .map(|&[_, falsy]| falsy),
        );
        falsy_visitor.visit_graph_inner(
            phi.falsy_mut(),
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

        let true_is_empty = branch_is_empty(phi.truthy(), phi.starts()[0]);
        let false_is_empty = branch_is_empty(phi.falsy(), phi.starts()[1]);

        if true_is_empty && false_is_empty {
            tracing::debug!("removing an empty phi {:?}", phi.node());

            graph.splice_ports(phi.effect_in(), phi.effect_out());

            for (&input_port, &param) in phi.inputs().iter().zip(phi.input_params()) {
                let truthy_param = phi.truthy().get_node(param[0]).to_input_param();
                let falsy_param = phi.falsy().get_node(param[1]).to_input_param();

                if let Some((Node::OutputPort(output), _, EdgeKind::Value)) =
                    phi.truthy().get_output(truthy_param.value())
                {
                    let output_port = phi.outputs().iter().zip(phi.output_params()).find_map(
                        |(&output_port, &param)| (param[0] == output.node()).then(|| output_port),
                    );

                    if let Some(output_port) = output_port {
                        tracing::debug!(
                            "splicing phi input to output passthrough {:?}->{:?}",
                            input_port,
                            output_port,
                        );

                        graph.splice_ports(input_port, output_port);
                    }
                } else if let Some((Node::OutputPort(output), _, EdgeKind::Value)) =
                    phi.falsy().get_output(falsy_param.value())
                {
                    let output_port = phi.outputs().iter().zip(phi.output_params()).find_map(
                        |(&output_port, &param)| (param[1] == output.node()).then(|| output_port),
                    );

                    if let Some(output_port) = output_port {
                        tracing::debug!(
                            "splicing phi input to output passthrough {:?}->{:?}",
                            input_port,
                            output_port,
                        );

                        graph.splice_ports(input_port, output_port);
                    }
                }
            }

            graph.remove_node(phi.node());
            self.changed();
        } else {
            graph.replace_node(phi.node(), phi);
        }
    }

    // TODO: Remove the theta node if it's unused?
    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut visitor = Self::new();

        self.stack_buf.extend(
            theta
                .input_params()
                .iter()
                .chain(theta.output_params())
                .copied(),
        );

        visitor.visit_graph_inner(
            theta.body_mut(),
            &mut self.stack_buf,
            &mut self.visited_buf,
            &mut self.buffer_buf,
        );
        self.changed |= visitor.did_change();

        // TODO: Buffer these
        let mut unused_input_ports = BTreeSet::new();
        let mut unused_input_params = BTreeSet::new();
        let mut unused_output_ports = BTreeSet::new();
        let mut unused_output_params = BTreeSet::new();

        for (&port, &param) in theta.inputs().iter().zip(theta.input_params()) {
            let input = theta.body().get_node(param).to_input_param();
            let param_output = theta.body().get_output(input.value());

            match param_output {
                // If the input param is unused, remove the port and parameter
                None => {
                    tracing::debug!(
                        "removing input param from theta node {:?}: {:?} with input param {:?}",
                        theta.node(),
                        port,
                        param,
                    );

                    unused_input_ports.insert(port);
                    unused_input_params.insert(param);

                    self.changed();
                }

                // TODO: allow removing effect edges through loops as well
                Some((consumer, _, EdgeKind::Value)) => {
                    if let Some(output) = consumer.as_output_param() {
                        let (&output_port, &output_param) = theta
                            .outputs()
                            .iter()
                            .zip(theta.output_params())
                            .find(|(_, &param)| param == output.node())
                            .unwrap();

                        tracing::debug!(
                            theta = ?theta.node(),
                            input_port = ?port,
                            input_param = ?param,
                            ?output_port,
                            ?output_param,
                            "theta input param is a passthrough, rerouting to its original value",
                        );

                        let (_, producer, _) = graph.get_input(port);
                        graph.rewire_dependents(output_port, producer);

                        unused_output_ports.insert(output_port);
                        unused_output_params.insert(output_param);

                        self.changed();
                    }
                }

                _ => {}
            }
        }

        // Remove the nodes from the subgraph
        for &param in unused_input_params.iter() {
            theta.body_mut().remove_node(param);
        }

        // Remove the unused ports and parameters from the theta node
        theta
            .inputs_mut()
            .retain(|port| !unused_input_ports.contains(port));
        theta
            .input_params_mut()
            .retain(|param| !unused_input_params.contains(param));

        for (&port, &param) in theta.outputs().iter().zip(theta.output_params()) {
            let output = theta.body().get_node(param).to_output_param();

            // If the output param is unused, grab the port and parameter
            let port_is_unused = theta.body().try_input(output.value()).is_none()
                || graph.get_output(port).is_none();

            if port_is_unused {
                tracing::debug!(
                    "removing output param from theta node {:?}: {:?} with input output {:?}",
                    theta.node(),
                    port,
                    param,
                );

                unused_output_ports.insert(port);
                unused_output_params.insert(param);

                self.changed();
            }
        }

        // Remove the nodes from the subgraph
        for &param in unused_output_params.iter() {
            theta.body_mut().remove_node(param);
        }

        // Remove the unused ports and parameters from the theta node
        theta
            .outputs_mut()
            .retain(|port| !unused_output_ports.contains(port));
        theta
            .output_params_mut()
            .retain(|param| !unused_output_params.contains(param));

        graph.replace_node(theta.node(), theta);
    }

    // TODO: Remove unused {Input, Output}Params
}

impl Default for Dce {
    fn default() -> Self {
        Self::new()
    }
}
