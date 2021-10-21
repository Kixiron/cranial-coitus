use crate::{
    graph::{Add, EdgeKind, Eq, Gamma, Load, Neg, Node, NodeId, Not, Rvsdg, Store, Theta},
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
            let node = graph.get_node(node_id);
            // let input_descriptor = node.input_desc();
            // let output_descriptor = node.output_desc();
            //
            // let value_inputs = graph.value_input_count(node_id);
            // let effect_inputs = graph.effect_input_count(node_id);
            //
            // let value_outputs = graph.value_output_count(node_id);
            // let effect_outputs = graph.effect_output_count(node_id);

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
            // || ((!input_descriptor.effect().contains(effect_inputs)
            //     || !input_descriptor.value().contains(value_inputs)
            //     || !output_descriptor.effect().contains(effect_outputs)
            //     || !output_descriptor.value().contains(value_outputs))
            //     && !node.is_start()
            //     && !node.is_end()
            //     && !node.is_input_port()
            //     && !node.is_output_port())
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

    fn visit_add(&mut self, graph: &mut Rvsdg, add: Add) {
        if graph.get_output(add.value()).is_none()
            || graph.try_input(add.lhs()).is_none()
            || graph.try_input(add.rhs()).is_none()
        {
            tracing::debug!(
                node = ?graph.get_node(add.node()),
                "removed dead add {:?}",
                add.node(),
            );

            graph.remove_node(add.node());
        }
    }

    fn visit_eq(&mut self, graph: &mut Rvsdg, eq: Eq) {
        if graph.get_output(eq.value()).is_none()
            || graph.try_input(eq.lhs()).is_none()
            || graph.try_input(eq.rhs()).is_none()
        {
            tracing::debug!(
                node = ?graph.get_node(eq.node()),
                "removed dead eq {:?}",
                eq.node(),
            );

            graph.remove_node(eq.node());
        }
    }

    fn visit_not(&mut self, graph: &mut Rvsdg, not: Not) {
        if graph.get_output(not.value()).is_none() || graph.try_input(not.input()).is_none() {
            tracing::debug!(
                node = ?graph.get_node(not.node()),
                "removed dead not {:?}",
                not.node(),
            );

            graph.remove_node(not.node());
        }
    }

    fn visit_neg(&mut self, graph: &mut Rvsdg, neg: Neg) {
        if graph.get_output(neg.value()).is_none() || graph.try_input(neg.input()).is_none() {
            tracing::debug!(
                node = ?graph.get_node(neg.node()),
                "removed dead neg {:?}",
                neg.node(),
            );

            graph.remove_node(neg.node());
        }
    }

    fn visit_store(&mut self, graph: &mut Rvsdg, store: Store) {
        if graph.get_output(store.effect()).is_none()
            || graph.try_input(store.effect_in()).is_none()
            || graph.try_input(store.ptr()).is_none()
            || graph.try_input(store.value()).is_none()
        {
            tracing::debug!(
                node = ?graph.get_node(store.node()),
                "removed dead store {:?}",
                store.node(),
            );

            graph.remove_node(store.node());
        }
    }

    fn visit_load(&mut self, graph: &mut Rvsdg, load: Load) {
        if graph.get_output(load.effect()).is_none()
            || graph.get_output(load.value()).is_none()
            || graph.try_input(load.effect_in()).is_none()
            || graph.try_input(load.ptr()).is_none()
        {
            tracing::debug!(
                node = ?graph.get_node(load.node()),
                "removed dead load {:?}",
                load.node(),
            );

            graph.remove_node(load.node());
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
                    gamma.true_branch().get_output(truthy_param.value())
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
                    gamma.false_branch().get_output(falsy_param.value())
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
