use crate::{
    graph::{
        Bool, EdgeKind, End, Gamma, InputParam, InputPort, Node, NodeExt, NodeId, OutputParam,
        OutputPort, Rvsdg, Start, Theta,
    },
    passes::{utils::ConstantStore, Pass},
    utils::{AssertNone, HashMap},
};

/// Evaluates constant operations within the program
// TODO: Write this within an iterative style so that
//       we can handle deeply nested programs
pub struct ElimConstGamma {
    values: ConstantStore,
    nodes: Vec<(NodeId, Node)>,
    input_lookup: HashMap<InputPort, Vec<InputPort>>,
    output_lookup: HashMap<OutputPort, OutputPort>,
    changed: bool,
}

impl ElimConstGamma {
    pub fn new(tape_len: u16) -> Self {
        Self {
            values: ConstantStore::new(tape_len),
            nodes: Vec::new(),
            input_lookup: HashMap::default(),
            output_lookup: HashMap::default(),
            changed: false,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }
}

impl Pass for ElimConstGamma {
    fn pass_name(&self) -> &'static str {
        "eliminate-const-gamma"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.values.clear();
        self.nodes.clear();
        self.input_lookup.clear();
        self.output_lookup.clear();
        self.changed = false;
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        self.values.add(bool.value(), value);
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let (mut truthy_visitor, mut falsy_visitor) = (
            Self::new(self.values.tape_len()),
            Self::new(self.values.tape_len()),
        );

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[true_param, false_param]) in gamma.inputs().iter().zip(gamma.input_params())
        {
            let (_, source, _) = graph.get_input(input);

            if let Some(constant) = self.values.get(source) {
                let true_param = gamma.true_branch().to_node::<InputParam>(true_param);
                truthy_visitor.values.add(true_param.output(), constant);

                let false_param = gamma.true_branch().to_node::<InputParam>(false_param);
                falsy_visitor.values.add(false_param.output(), constant);
            }
        }

        // If a constant condition is found, inline the chosen branch
        // Note that we don't actually visit the inlined subgraph in this iteration,
        // we leave that for successive passes to take care of
        if let Some(condition) = self.values.bool(graph.get_input(gamma.condition()).1) {
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

            debug_assert!(
                self.nodes.is_empty()
                    && self.input_lookup.is_empty()
                    && self.output_lookup.is_empty()
            );
            self.nodes.extend(
                chosen_branch
                    .iter_nodes()
                    .map(|(node_id, node)| (node_id, node.clone())),
            );

            for (&input, &params) in gamma.inputs().iter().zip(gamma.input_params()) {
                let param = if condition { params[0] } else { params[1] };
                let input_param = chosen_branch.to_node::<InputParam>(param);
                let inlined_output = graph.input_source(input);

                self.output_lookup
                    .insert(input_param.output(), inlined_output)
                    .debug_unwrap_none();
            }

            for (&output, &params) in gamma.outputs().iter().zip(gamma.output_params()) {
                let param = if condition { params[0] } else { params[1] };
                let output_param = chosen_branch.to_node::<OutputParam>(param);

                self.input_lookup
                    .insert(
                        output_param.input(),
                        graph.output_dest(output).collect::<Vec<_>>(),
                    )
                    .debug_unwrap_none();
            }

            // Inline the graph nodes, create the inlined ports and build the input/output maps
            for (node_id, mut node) in self.nodes.drain(..) {
                // Replace start nodes with the gamma's input effect
                if node.is_start() {
                    let starts = gamma.starts();
                    let start_id = if condition { starts[0] } else { starts[1] };
                    let start = chosen_branch.to_node::<Start>(start_id);

                    let output_effect = graph.input_source(gamma.input_effect());
                    self.output_lookup
                        .insert(start.effect(), output_effect)
                        .debug_unwrap_none();

                // Replace end nodes with the gamma's output effect
                } else if node.is_end() {
                    let ends = gamma.ends();
                    let end_id = if condition { ends[0] } else { ends[1] };
                    let end = chosen_branch.to_node::<End>(end_id);

                    self.input_lookup
                        .insert(
                            end.input_effect(),
                            graph.output_dest(gamma.output_effect()).collect(),
                        )
                        .debug_unwrap_none();

                // Ignore input and output ports
                // Otherwise just create the required ports & inline the node
                } else if !node.is_input_param() && !node.is_output_param() {
                    if let Some(theta) = node.as_theta_mut() {
                        if let Some(input_effect) = theta.input_effect() {
                            let inlined = graph.create_input_port(theta.node(), EdgeKind::Effect);

                            let replaced = self.input_lookup.insert(input_effect, vec![inlined]);
                            debug_assert!(
                                replaced.is_none() ,
                                "replaced effect {:?} for theta input effect port {:?} with {:?}\ninputs: {:?}\noutputs: {:?}",
                                replaced,
                                input_effect,
                                inlined,
                                self.input_lookup,
                                self.output_lookup,
                            );

                            theta.set_input_effect(inlined);
                        }

                        {
                            let mut new_invariant_inputs = HashMap::with_capacity_and_hasher(
                                theta.invariant_inputs_len(),
                                Default::default(),
                            );
                            for (port, param) in theta.invariant_input_pair_ids() {
                                let inlined =
                                    graph.create_input_port(theta.node(), EdgeKind::Value);

                                let replaced = self.input_lookup.insert(port, vec![inlined]);
                                debug_assert!(
                                    replaced.is_none() ,
                                    "replaced value {:?} for theta input port {:?} with {:?}\ninputs: {:?}\noutputs: {:?}",
                                    replaced,
                                    port,
                                    inlined,
                                    self.input_lookup,
                                    self.output_lookup,
                                );

                                new_invariant_inputs
                                    .insert(inlined, param)
                                    .debug_unwrap_none();
                            }

                            theta.replace_invariant_inputs(new_invariant_inputs);
                        }

                        {
                            let total_variant_inputs = theta.variant_inputs_len();
                            let (mut new_variant_inputs, mut new_output_feedback) = (
                                HashMap::with_capacity_and_hasher(
                                    total_variant_inputs,
                                    Default::default(),
                                ),
                                HashMap::with_capacity_and_hasher(
                                    total_variant_inputs,
                                    Default::default(),
                                ),
                            );

                            for (port, param, output) in
                                theta.variant_input_pair_ids_with_feedback()
                            {
                                let inlined =
                                    graph.create_input_port(theta.node(), EdgeKind::Value);

                                let replaced = self.input_lookup.insert(port, vec![inlined]);
                                debug_assert!(
                                    replaced.is_none() ,
                                    "replaced value {:?} for theta input port {:?} with {:?}\ninputs: {:?}\noutputs: {:?}",
                                    replaced,
                                    port,
                                    inlined,
                                    self.input_lookup,
                                    self.output_lookup,
                                );

                                new_variant_inputs
                                    .insert(inlined, param)
                                    .debug_unwrap_none();
                                new_output_feedback
                                    .insert(output, inlined)
                                    .debug_unwrap_none();
                            }

                            theta.replace_variant_inputs(new_variant_inputs);
                            theta.replace_output_feedback(new_output_feedback);
                        }

                        {
                            let total_outputs = theta.outputs_len();
                            let (mut new_outputs, mut new_output_feedback) = (
                                HashMap::with_capacity_and_hasher(
                                    total_outputs,
                                    Default::default(),
                                ),
                                HashMap::with_capacity_and_hasher(
                                    total_outputs,
                                    Default::default(),
                                ),
                            );

                            for (port, param, input) in theta.output_pair_ids_with_feedback() {
                                let inlined =
                                    graph.create_output_port(theta.node(), EdgeKind::Value);

                                let replaced = self.output_lookup.insert(port, inlined);
                                debug_assert!(
                                    replaced.is_none() ,
                                    "replaced value {:?} for theta output port {:?} with {:?}\ninputs: {:?}\noutputs: {:?}",
                                    replaced,
                                    port,
                                    inlined,
                                    self.input_lookup,
                                    self.output_lookup,
                                );

                                new_outputs.insert(inlined, param).debug_unwrap_none();
                                new_output_feedback
                                    .insert(inlined, input)
                                    .debug_unwrap_none();
                            }

                            theta.replace_outputs(new_outputs);
                            theta.replace_output_feedback(new_output_feedback);
                        }

                        if let Some(output_effect) = theta.output_effect() {
                            let inlined = graph.create_output_port(theta.node(), EdgeKind::Effect);

                            let replaced = self.output_lookup.insert(output_effect, inlined);
                            debug_assert!(
                                replaced.is_none() ,
                                "replaced effect {:?} for theta output port {:?} with {:?}\ninputs: {:?}\noutputs: {:?}",
                                replaced,
                                output_effect,
                                inlined,
                                self.input_lookup,
                                self.output_lookup,
                            );

                            theta.set_output_effect(inlined);
                        }
                    } else {
                        node.update_inputs(|input, kind| {
                            let inlined = graph.create_input_port(node_id, kind);
                            self.input_lookup
                                .insert(input, vec![inlined])
                                .debug_unwrap_none();

                            Some(inlined)
                        });

                        node.update_outputs(|output, kind| {
                            let inlined = graph.create_output_port(node_id, kind);
                            self.output_lookup
                                .insert(output, inlined)
                                .debug_unwrap_none();

                            Some(inlined)
                        });
                    }

                    graph.add_node(node_id, node);
                }
            }

            for node_id in chosen_branch.node_ids() {
                for (branch_input, _, branch_output, kind) in chosen_branch.all_node_inputs(node_id)
                {
                    if let Some(output) = self.output_lookup.get(&branch_output).copied() {
                        if let Some(inputs) = self.input_lookup.get(&branch_input) {
                            for &input in inputs {
                                graph.add_edge(output, input, kind);
                            }
                        }
                    } else {
                        tracing::error!(
                            "failed to add edge while inlining gamma branch, missing output port for {:?}",
                            branch_output,
                        );
                    }
                }
            }

            self.nodes.clear();
            self.input_lookup.clear();
            self.output_lookup.clear();

            graph.remove_node(gamma.node());
            self.changed();

        // If we can't find a constant condition for this gamma, just visit all of its branches
        } else {
            let mut changed = false;
            changed |= truthy_visitor.visit_graph(gamma.true_mut());
            changed |= falsy_visitor.visit_graph(gamma.false_mut());

            if changed {
                graph.replace_node(gamma.node(), gamma);
                self.changed();
            }
        }
    }

    // FIXME: Inlining gamma bodies is broken rn
    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut changed = false;
        let mut visitor = Self::new(self.values.tape_len());

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (input, param) in theta.invariant_input_pairs() {
            if let Some(constant) = self.values.get(graph.input_source(input)) {
                visitor.values.add(param.output(), constant);
            }
        }

        changed |= visitor.visit_graph(theta.body_mut());

        let cond_out = theta.condition();
        let cond_source = theta.body().input_source(cond_out.input());
        let cond_value = visitor.values.bool(cond_source);

        // If the theta's condition is `false`, it will never loop and we can inline the body
        if cond_value == Some(false) {
            tracing::debug!(
                "eliminated theta {:?} with false as its loop variable, inlining its body once",
                theta.node(),
            );

            // TODO: Factor subgraph inlining into a function
            debug_assert!(
                self.nodes.is_empty()
                    && self.input_lookup.is_empty()
                    && self.output_lookup.is_empty()
            );
            self.nodes.extend(
                theta
                    .body()
                    .iter_nodes()
                    .map(|(node_id, node)| (node_id, node.clone())),
            );

            for (input, param) in theta.input_pairs() {
                let inlined_output = graph.input_source(input);

                self.output_lookup
                    .insert(param.output(), inlined_output)
                    .debug_unwrap_none();
            }

            for (output, param) in theta.output_pairs() {
                self.input_lookup
                    .insert(param.input(), graph.output_dest(output).collect::<Vec<_>>())
                    .debug_unwrap_none();
            }

            // Inline the graph nodes, create the inlined ports and build the input/output maps
            for (node_id, mut node) in self.nodes.drain(..) {
                // Replace start nodes with the gamma's input effect
                if node.is_start() {
                    let start = theta.start_node();
                    let output_effect = graph.input_source(theta.input_effect().unwrap());

                    self.output_lookup
                        .insert(start.effect(), output_effect)
                        .debug_unwrap_none();

                // Replace end nodes with the gamma's output effect
                } else if node.is_end() {
                    let end = theta.end_node();
                    self.input_lookup
                        .insert(
                            end.input_effect(),
                            graph.output_dest(theta.output_effect().unwrap()).collect(),
                        )
                        .debug_unwrap_none();

                // Ignore input and output ports
                // Otherwise just create the required ports & inline the node
                } else if !node.is_input_param() && !node.is_output_param() {
                    if let Some(theta) = node.as_theta_mut() {
                        if let Some(input_effect) = theta.input_effect() {
                            let inlined = graph.create_input_port(theta.node(), EdgeKind::Effect);

                            let replaced = self.input_lookup.insert(input_effect, vec![inlined]);
                            debug_assert!(
                                replaced.is_none() ,
                                "replaced effect {:?} for theta input effect port {:?} with {:?}\ninputs: {:?}\noutputs: {:?}",
                                replaced,
                                input_effect,
                                inlined,
                                self.input_lookup,
                                self.output_lookup,
                            );

                            theta.set_input_effect(inlined);
                        }

                        {
                            let total_invariant_inputs = theta.invariant_inputs_len();
                            let mut new_invariant_inputs = HashMap::with_capacity_and_hasher(
                                total_invariant_inputs,
                                Default::default(),
                            );

                            for (port, param) in theta.invariant_input_pair_ids() {
                                let inlined =
                                    graph.create_input_port(theta.node(), EdgeKind::Value);

                                let replaced = self.input_lookup.insert(port, vec![inlined]);
                                debug_assert!(
                                    replaced.is_none() ,
                                    "replaced value {:?} for theta input port {:?} with {:?}\ninputs: {:?}\noutputs: {:?}",
                                    replaced,
                                    port,
                                    inlined,
                                    self.input_lookup,
                                    self.output_lookup,
                                );

                                new_invariant_inputs
                                    .insert(inlined, param)
                                    .debug_unwrap_none();
                            }

                            theta.replace_invariant_inputs(new_invariant_inputs);
                        }

                        {
                            let total_variant_inputs = theta.variant_inputs_len();
                            let (mut new_variant_inputs, mut new_output_feedback) = (
                                HashMap::with_capacity_and_hasher(
                                    total_variant_inputs,
                                    Default::default(),
                                ),
                                HashMap::with_capacity_and_hasher(
                                    total_variant_inputs,
                                    Default::default(),
                                ),
                            );

                            for (port, param, output) in
                                theta.variant_input_pair_ids_with_feedback()
                            {
                                let inlined =
                                    graph.create_input_port(theta.node(), EdgeKind::Value);

                                let replaced = self.input_lookup.insert(port, vec![inlined]);
                                debug_assert!(
                                    replaced.is_none() ,
                                    "replaced value {:?} for theta input port {:?} with {:?}\ninputs: {:?}\noutputs: {:?}",
                                    replaced,
                                    port,
                                    inlined,
                                    self.input_lookup,
                                    self.output_lookup,
                                );

                                new_variant_inputs
                                    .insert(inlined, param)
                                    .debug_unwrap_none();
                                new_output_feedback
                                    .insert(output, inlined)
                                    .debug_unwrap_none();
                            }

                            theta.replace_variant_inputs(new_variant_inputs);
                            theta.replace_output_feedback(new_output_feedback);
                        }

                        {
                            let total_outputs = theta.outputs_len();
                            let (mut new_outputs, mut new_output_feedback) = (
                                HashMap::with_capacity_and_hasher(
                                    total_outputs,
                                    Default::default(),
                                ),
                                HashMap::with_capacity_and_hasher(
                                    total_outputs,
                                    Default::default(),
                                ),
                            );

                            for (port, param, input) in theta.output_pair_ids_with_feedback() {
                                let inlined =
                                    graph.create_output_port(theta.node(), EdgeKind::Value);

                                let replaced = self.output_lookup.insert(port, inlined);
                                debug_assert!(
                                    replaced.is_none() ,
                                    "replaced value {:?} for theta output port {:?} with {:?}\ninputs: {:?}\noutputs: {:?}",
                                    replaced,
                                    port,
                                    inlined,
                                    self.input_lookup,
                                    self.output_lookup,
                                );

                                new_outputs.insert(inlined, param).debug_unwrap_none();
                                new_output_feedback
                                    .insert(inlined, input)
                                    .debug_unwrap_none();
                            }

                            theta.replace_outputs(new_outputs);
                            theta.replace_output_feedback(new_output_feedback);
                        }

                        if let Some(output_effect) = theta.output_effect() {
                            let inlined = graph.create_output_port(theta.node(), EdgeKind::Effect);

                            let replaced = self.output_lookup.insert(output_effect, inlined);
                            debug_assert!(
                                replaced.is_none() ,
                                "replaced effect {:?} for theta output port {:?} with {:?}\ninputs: {:?}\noutputs: {:?}",
                                replaced,
                                output_effect,
                                inlined,
                                self.input_lookup,
                                self.output_lookup,
                            );

                            theta.set_output_effect(inlined);
                        }
                    } else {
                        node.update_inputs(|input, kind| {
                            let inlined = graph.create_input_port(node_id, kind);
                            self.input_lookup
                                .insert(input, vec![inlined])
                                .debug_unwrap_none();

                            Some(inlined)
                        });

                        node.update_outputs(|output, kind| {
                            let inlined = graph.create_output_port(node_id, kind);
                            self.output_lookup
                                .insert(output, inlined)
                                .debug_unwrap_none();

                            Some(inlined)
                        });
                    }

                    graph.add_node(node_id, node);
                }
            }

            for node_id in theta.body().node_ids() {
                for (branch_input, _, branch_output, kind) in theta.body().all_node_inputs(node_id)
                {
                    let (inputs, output) = (
                        self.input_lookup.get(&branch_input).into_iter().flatten(),
                        self.output_lookup.get(&branch_output).copied(),
                    );

                    if let Some(output) = output {
                        for &input in inputs {
                            graph.add_edge(output, input, kind);
                        }
                    } else {
                        tracing::error!(
                            "failed to add {} edge while inlining theta branch, missing output port for {:?} (inputs: {:?})",
                            kind,
                            branch_output,
                            inputs.collect::<Vec<_>>(),
                        );
                    }
                }
            }

            self.nodes.clear();
            self.input_lookup.clear();
            self.output_lookup.clear();

            graph.remove_node(theta.node());
            self.changed();
        } else if changed {
            graph.replace_node(theta.node(), theta);
            self.changed();
        }
    }
}
