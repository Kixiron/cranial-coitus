//! The theta node

use crate::{
    graph::{
        nodes::node_ext::{InputPortKinds, InputPorts, OutputPortKinds, OutputPorts},
        EdgeCount, EdgeDescriptor, EdgeKind, End, InputParam, InputPort, NodeExt, NodeId,
        OutputParam, OutputPort, Rvsdg, Start, Subgraph,
    },
    utils::{AssertNone, HashMap},
};
use tinyvec::TinyVec;

// TODO: Probably want reverse lookup maps as well as the current forward ones
#[derive(Debug, Clone, PartialEq)]
pub struct Theta {
    /// The theta's [`NodeId`]
    node: NodeId,

    /// Theta nodes don't have to have effects, the inner [`Subgraph`]
    /// will have them regardless (only from [`Start`] to [`End`] node though).
    /// Just because a theta node isn't effectful doesn't mean it's useless,
    /// it can still have data dependencies
    ///
    /// [`Start`]: crate::graph::Start
    /// [`End`]: crate::graph::End
    effects: Option<ThetaEffects>,

    /// These are the inputs that go into the loop's body
    /// but do not change between iterations
    ///
    /// These point from an [`InputPort`] on the theta to the [`NodeId`]
    /// of an [`InputParam`] within the theta's body
    invariant_inputs: HashMap<InputPort, NodeId>,

    /// These inputs change upon each iteration, they're connected to `output_back_edges`
    ///
    /// These point from an [`InputPort`] on the theta to the [`NodeId`]
    /// of an [`InputParam`] within the theta's body
    variant_inputs: HashMap<InputPort, NodeId>,

    /// The node's outputs, these all must be included in `output_back_edges`. Any invariant
    /// data must go around the loop to reach dependents after it
    ///
    /// These point from an [`OutputPort`] on the theta to the [`NodeId`]
    /// of an [`OutputParam`] within the theta's body
    outputs: HashMap<OutputPort, NodeId>,

    /// These are the relationships between outputs and variant inputs
    ///
    /// These point from an [`OutputPort`] in `outputs` to an [`InputPort`] in `variant_inputs`
    output_feedback: HashMap<OutputPort, InputPort>,

    /// The theta's condition, it should be an expression that evaluates to a boolean
    ///
    /// Points to an [`OutputParam`] within the theta's body
    condition: NodeId,

    /// The theta's loop body, contains the [`Start`] and [`End`] [`NodeId`]s
    ///
    /// [`Start`]: crate::graph::Start
    /// [`End`]: crate::graph::End
    subgraph: Subgraph,
}

impl Theta {
    /// Create a new theta node
    #[allow(clippy::too_many_arguments)]
    pub(in crate::graph) fn new(
        node: NodeId,
        effects: Option<ThetaEffects>,
        invariant_inputs: HashMap<InputPort, NodeId>,
        variant_inputs: HashMap<InputPort, NodeId>,
        outputs: HashMap<OutputPort, NodeId>,
        output_feedback: HashMap<OutputPort, InputPort>,
        condition: NodeId,
        subgraph: Subgraph,
    ) -> Self {
        if cfg!(debug_assertions) {
            assert_eq!(variant_inputs.len(), outputs.len());
            assert_eq!(outputs.len(), output_feedback.len());

            for (output, input) in &output_feedback {
                assert!(outputs.contains_key(output));
                assert!(variant_inputs.contains_key(input));
                assert!(!invariant_inputs.contains_key(input));
            }

            for input in variant_inputs.keys() {
                assert!(!invariant_inputs.contains_key(input));
            }
        }

        Self {
            node,
            effects,
            invariant_inputs,
            variant_inputs,
            outputs,
            output_feedback,
            condition,
            subgraph,
        }
    }

    /// Get the [`NodeId`] of the theta node's condition
    ///
    /// Should be an [`OutputParam`] in the theta's body
    pub const fn condition_id(&self) -> NodeId {
        self.condition
    }

    /// Get the [`OutputParam`] of the theta's condition from within its body
    pub fn condition(&self, graph: &Rvsdg) -> OutputParam {
        *graph.to_node(self.condition)
    }

    /// Get the [`Start`] of the theta's body
    pub fn start_node(&self, graph: &Rvsdg) -> Start {
        self.subgraph.start_node(graph)
    }

    /// Get the [`End`] of the theta's body
    pub fn end_node(&self, graph: &Rvsdg) -> End {
        self.subgraph.end_node(graph)
    }

    pub const fn start_id(&self) -> NodeId {
        self.subgraph.start
    }

    pub const fn end_id(&self) -> NodeId {
        self.subgraph.end
    }

    /// Get the [`NodeId`] of the theta body's [`End`] node
    pub const fn end_node_id(&self) -> NodeId {
        self.subgraph.end
    }

    /// Get the input effect's port from the theta node if it's available
    pub fn input_effect(&self) -> Option<InputPort> {
        self.effects.map(|effects| effects.input)
    }

    /// Get a mutable reference to the input effect's port from
    /// the theta node if it's available
    pub fn input_effect_mut(&mut self) -> Option<&mut InputPort> {
        self.effects.as_mut().map(|effects| &mut effects.input)
    }

    /// Get the output effect's port from the theta node if available
    pub fn output_effect(&self) -> Option<OutputPort> {
        self.effects.map(|effects| effects.output)
    }

    /// Get a mutable reference to the output effect's port from
    /// the theta node if it's available
    pub fn output_effect_mut(&mut self) -> Option<&mut OutputPort> {
        self.effects.as_mut().map(|effects| &mut effects.output)
    }

    pub fn effects(&self) -> Option<(InputPort, OutputPort)> {
        self.effects.map(|effects| (effects.input, effects.output))
    }

    pub const fn has_effects(&self) -> bool {
        self.effects.is_some()
    }

    /// Get the theta's output feedback edges
    pub const fn output_feedback(&self) -> &HashMap<OutputPort, InputPort> {
        &self.output_feedback
    }

    pub fn set_input_effect(&mut self, input_effect: InputPort) {
        if let Some(effects) = self.effects.as_mut() {
            effects.input = input_effect;
        } else {
            tracing::error!("tried to set input effect on theta without effect edges");
        }
    }

    pub fn set_output_effect(&mut self, output_effect: OutputPort) {
        if let Some(effects) = self.effects.as_mut() {
            effects.output = output_effect;
        } else {
            tracing::error!("tried to set output effect on theta without effect edges");
        }
    }

    pub fn remove_invariant_input(&mut self, input: InputPort) {
        self.invariant_inputs.remove(&input);
    }

    pub fn add_invariant_input_raw(&mut self, input: InputPort, param: NodeId) {
        // debug_assert!(graph.contains_node(param));
        // debug_assert!(graph.get_node(param).is_input_param());

        self.invariant_inputs
            .insert(input, param)
            .debug_unwrap_none();
    }

    pub fn contains_invariant_input(&self, input: InputPort) -> bool {
        self.invariant_inputs.contains_key(&input)
    }

    pub fn contains_variant_input(&self, input: InputPort) -> bool {
        self.variant_inputs.contains_key(&input)
    }

    /// Returns `true` if the given [`InputParam`] is a variant input on the current
    /// [`Theta`] node
    pub fn contains_variant_input_param(&self, param: &InputParam) -> bool {
        self.variant_inputs
            .values()
            .any(|&node| node == param.node())
    }

    pub fn retain_invariant_inputs<F>(&mut self, mut retain: F)
    where
        F: FnMut(InputPort, NodeId) -> bool,
    {
        self.invariant_inputs
            .retain(|&port, &mut param| retain(port, param));
    }

    /// Removes a variant input, the output it's fed from and the feedback entry for them
    pub fn remove_variant_input(&mut self, input: InputPort) {
        self.variant_inputs.remove(&input).debug_unwrap();

        let output = self
            .output_feedback
            .iter()
            .find_map(|(&output, &port)| (port == input).then(|| output))
            .unwrap();

        self.output_feedback.remove(&output).debug_unwrap();
        self.outputs.remove(&output).debug_unwrap();
    }

    /// Returns the number of all inputs (variant and invariant) to the theta node,
    /// *not including effect inputs*
    pub fn inputs_len(&self) -> usize {
        self.invariant_inputs_len() + self.variant_inputs_len()
    }

    /// Returns the input ports of all inputs (variant and invariant) to the theta node,
    /// *not including effect inputs*
    pub fn input_ports(&self) -> impl Iterator<Item = InputPort> + '_ {
        self.invariant_input_ports()
            .chain(self.variant_input_ports())
    }

    /// Returns the input ports and the associated input param of all inputs (variant and invariant)
    /// to the theta node, *not including effect inputs*
    pub fn input_pairs<'a>(
        &'a self,
        graph: &'a Rvsdg,
    ) -> impl Iterator<Item = (InputPort, InputParam)> + 'a {
        self.invariant_input_pairs(graph)
            .chain(self.variant_input_pairs(graph))
    }

    /// Returns the input ports and the associated node id of each input param for all inputs (variant and invariant)
    /// to the theta node, *not including effect inputs*
    pub fn input_pair_ids(&self) -> impl Iterator<Item = (InputPort, NodeId)> + '_ {
        self.invariant_input_pair_ids()
            .chain(self.variant_input_pair_ids())
    }

    /// Returns all input params to the theta node, *not including effect inputs*
    pub fn input_params<'a>(&'a self, graph: &'a Rvsdg) -> impl Iterator<Item = InputParam> + 'a {
        self.invariant_input_params(graph)
            .chain(self.variant_input_params(graph))
    }

    /// Returns the node ids of all input params to the theta node (variant and invariant),
    /// *not including effect inputs*
    pub fn input_param_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.invariant_input_param_ids()
            .chain(self.variant_input_param_ids())
    }

    /// Get the number of invariant inputs to the theta node
    pub fn invariant_inputs_len(&self) -> usize {
        self.invariant_inputs.len()
    }

    /// Returns the invariant inputs to the theta node
    pub fn invariant_input_ports(&self) -> impl Iterator<Item = InputPort> + '_ {
        self.invariant_inputs.keys().copied()
    }

    /// Returns the invariant inputs and the input ports that feed into them
    pub fn invariant_input_pairs<'a>(
        &'a self,
        graph: &'a Rvsdg,
    ) -> impl Iterator<Item = (InputPort, InputParam)> + 'a {
        self.invariant_inputs
            .iter()
            .map(|(&port, &param)| (port, *graph.to_node(param)))
    }

    /// Returns the node ids of each invariant input and the input port that feeds into them
    pub fn invariant_input_pair_ids(&self) -> impl Iterator<Item = (InputPort, NodeId)> + '_ {
        self.invariant_inputs
            .iter()
            .map(|(&port, &param)| (port, param))
    }

    /// Returns the invariant inputs to the theta node
    pub fn invariant_input_params<'a>(
        &'a self,
        graph: &'a Rvsdg,
    ) -> impl Iterator<Item = InputParam> + 'a {
        self.invariant_inputs
            .iter()
            .map(|(_, &param)| *graph.to_node(param))
    }

    /// Returns the node ids of invariant inputs to the theta node
    pub fn invariant_input_param_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.invariant_inputs.values().copied()
    }

    pub fn replace_invariant_inputs(&mut self, invariant_inputs: HashMap<InputPort, NodeId>) {
        self.invariant_inputs = invariant_inputs;
    }

    /// Get the number of variant inputs to the theta node
    pub fn variant_inputs_len(&self) -> usize {
        self.variant_inputs.len()
    }

    /// Returns the variant inputs to the theta node
    pub fn variant_input_ports(&self) -> impl Iterator<Item = InputPort> + '_ {
        self.variant_inputs.keys().copied()
    }

    /// Returns the variant inputs and the input ports that feed into them
    pub fn variant_input_pairs<'a>(
        &'a self,
        graph: &'a Rvsdg,
    ) -> impl Iterator<Item = (InputPort, InputParam)> + 'a {
        self.variant_inputs
            .iter()
            .map(|(&port, &param)| (port, *graph.to_node(param)))
    }

    /// Returns the node ids of each variant input and the input port that feeds into them
    pub fn variant_input_pair_ids(&self) -> impl Iterator<Item = (InputPort, NodeId)> + '_ {
        self.variant_inputs
            .iter()
            .map(|(&port, &param)| (port, param))
    }

    /// Returns the variant inputs to the theta node
    pub fn variant_input_params<'a>(
        &'a self,
        graph: &'a Rvsdg,
    ) -> impl Iterator<Item = InputParam> + 'a {
        self.variant_inputs
            .iter()
            .map(|(_, &param)| *graph.to_node(param))
    }

    /// Returns the node ids of variant inputs to the theta node
    pub fn variant_input_param_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.variant_inputs.values().copied()
    }

    pub fn variant_input_pair_ids_with_feedback(
        &self,
    ) -> impl Iterator<Item = (InputPort, NodeId, OutputPort)> + '_ {
        self.variant_inputs.iter().map(|(&port, &param)| {
            (
                port,
                param,
                self.output_feedback
                    .iter()
                    .find_map(|(&output, &input)| (input == port).then(|| output))
                    .unwrap(),
            )
        })
    }

    pub fn replace_variant_inputs(&mut self, variant_inputs: HashMap<InputPort, NodeId>) {
        self.variant_inputs = variant_inputs;
    }

    /// Get the number of outputs from the theta node, *not including effect outputs*
    pub fn outputs_len(&self) -> usize {
        self.outputs.len()
    }

    /// Returns all output ports from the theta node, *not including effect outputs*
    pub fn output_ports(&self) -> impl Iterator<Item = OutputPort> + '_ {
        self.outputs.keys().copied()
    }

    /// Returns all output ports and the output params that they feed into,
    /// *not including effect outputs*
    pub fn output_pairs<'a>(
        &'a self,
        graph: &'a Rvsdg,
    ) -> impl Iterator<Item = (OutputPort, OutputParam)> + 'a {
        self.outputs
            .iter()
            .map(|(&port, &param)| (port, *graph.to_node(param)))
    }

    /// Returns the node ids of each output param and the output port that feeds into them
    pub fn output_pair_ids(&self) -> impl Iterator<Item = (OutputPort, NodeId)> + '_ {
        self.outputs.iter().map(|(&port, &param)| (port, param))
    }

    pub fn output_pair_ids_with_feedback(
        &self,
    ) -> impl Iterator<Item = (OutputPort, NodeId, InputPort)> + '_ {
        self.outputs
            .iter()
            .map(|(&port, &param)| (port, param, self.output_feedback[&port]))
    }

    pub fn replace_outputs(&mut self, outputs: HashMap<OutputPort, NodeId>) {
        self.outputs = outputs;
    }

    pub fn replace_output_feedback(&mut self, output_feedback: HashMap<OutputPort, InputPort>) {
        self.output_feedback = output_feedback;
    }

    /// Returns the outputs from the theta node, *not including effect outputs*
    pub fn output_params<'a>(&'a self, graph: &'a Rvsdg) -> impl Iterator<Item = OutputParam> + 'a {
        self.outputs.iter().map(|(_, &param)| *graph.to_node(param))
    }

    /// Returns the node ids of outputs from the theta node, *not including effect outputs*
    pub fn output_param_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.outputs.values().copied()
    }

    /// Returns all variant inputs to the theta node along with the output
    /// that loops back into the given input
    pub fn variant_inputs_loopback<'a>(
        &'a self,
        graph: &'a Rvsdg,
    ) -> impl Iterator<Item = (InputParam, OutputParam)> + 'a {
        self.output_feedback.iter().map(|(output, input)| {
            let (input, output) = (self.variant_inputs[input], self.outputs[output]);
            (*graph.to_node(input), *graph.to_node(output))
        })
    }

    /// Returns `true` if the given [`OutputParam`] feeds back to the given variant [`InputParam`]
    ///
    /// Will return `false` if either of the given [`OutputParam`] or [`InputParam`]s don't exist
    /// within the current [`Theta`] or if the given [`InputParam`] isn't a *variant* input.
    pub(crate) fn output_feeds_back_to(
        &self,
        output: &OutputParam,
        variant_input: &InputParam,
    ) -> bool {
        self.outputs
            .iter()
            .find(|(_, &output_id)| output_id == output.node())
            .and_then(|(output_port, _)| self.output_feedback.get(output_port))
            .and_then(|input_port| self.variant_inputs.get(input_port))
            .map_or(false, |&input_node| input_node == variant_input.node())
    }

    /// Gets the [`InputPort`] of the parent [`Theta`] that feeds into the given variant [`InputParam`]
    pub fn variant_input_source(&self, variant_input: &InputParam) -> Option<InputPort> {
        self.variant_inputs
            .iter()
            .find(|(_, &input)| input == variant_input.node())
            .map(|(&input_port, _)| input_port)
    }

    pub(crate) fn has_child_thetas(&self, graph: &Rvsdg) -> bool {
        let params = [self.start_id(), self.end_id()]
            .into_iter()
            .chain(self.input_param_ids())
            .chain(self.output_param_ids());

        graph.try_for_each_transitive_node_inner(params, |_, node| node.is_theta())
    }
}

/// Utility functions
// TODO: Function to inline the theta's body into the given graph
impl Theta {
    pub fn get_output_param(&self, port: OutputPort, graph: &Rvsdg) -> Option<OutputParam> {
        self.outputs.get(&port).map(|&node| *graph.to_node(node))
    }

    /// Returns `true` if the theta's condition is always `false`
    pub fn is_infinite(&self, graph: &Rvsdg) -> bool {
        let cond = self.condition(graph);
        let condition_is_false = graph
            .input_source_node(cond.input())
            .as_bool_value()
            .unwrap_or(false);

        condition_is_false
    }
}

impl NodeExt for Theta {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::new(
            EdgeCount::exact(self.input_effect().is_some() as usize),
            EdgeCount::exact(self.inputs_len()),
        )
    }

    fn all_input_ports(&self) -> InputPorts {
        let mut inputs =
            TinyVec::with_capacity(self.inputs_len() + self.input_effect().is_some() as usize);

        inputs.extend(self.input_ports());
        if let Some(input_effect) = self.input_effect() {
            inputs.push(input_effect);
        }

        inputs
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        let mut inputs =
            TinyVec::with_capacity(self.inputs_len() + self.input_effect().is_some() as usize);

        inputs.extend(self.input_ports().map(|input| (input, EdgeKind::Value)));
        if let Some(input_effect) = self.input_effect() {
            inputs.push((input_effect, EdgeKind::Effect));
        }

        inputs
    }

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        let parent_node = self.node;

        // Try to replace the input effect
        if let Some(input_effect) = self
            .input_effect_mut()
            .filter(|&&mut effect| effect == from)
        {
            tracing::trace!(
                node = ?parent_node,
                "replaced input effect {:?} of Theta with {:?}",
                from, to,
            );

            *input_effect = to;

        // Try to replace the invariant effects
        } else if let Some(node_id) = {
            let mut input_node = None;
            self.invariant_inputs.retain(|&input, &mut node| {
                if input == from {
                    debug_assert!(input_node.is_none());
                    input_node = Some(node);

                    false
                } else {
                    true
                }
            });

            input_node
        } {
            tracing::trace!(
                node = ?parent_node,
                "replaced invariant input effect {:?} of Theta with {:?}",
                from, to,
            );

            self.invariant_inputs
                .insert(to, node_id)
                .debug_unwrap_none();

        // Try to replace the variant effects
        } else if let Some(node_id) = {
            let mut input_node = None;
            self.variant_inputs.retain(|&input, &mut node| {
                if input == from {
                    debug_assert!(input_node.is_none());
                    input_node = Some(node);

                    false
                } else {
                    true
                }
            });

            input_node
        } {
            tracing::trace!(
                node = ?parent_node,
                "replaced variant input effect {:?} of Theta with {:?}",
                from, to,
            );

            self.variant_inputs.insert(to, node_id).debug_unwrap_none();
            for feedback in self.output_feedback.values_mut() {
                if *feedback == from {
                    *feedback = to;
                }
            }

        // Otherwise the theta doesn't have this input port
        } else {
            tracing::trace!(
                node = ?parent_node,
                "tried to replace input effect {:?} of Theta with {:?} but Theta doesn't have that port",
                from, to,
            );
        }
    }

    fn output_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::new(
            EdgeCount::exact(self.output_effect().is_some() as usize),
            EdgeCount::new(None, Some(self.outputs_len())),
        )
    }

    fn all_output_ports(&self) -> OutputPorts {
        let mut outputs =
            TinyVec::with_capacity(self.outputs_len() + self.output_effect().is_some() as usize);

        outputs.extend(self.output_ports());
        if let Some(output_effect) = self.output_effect() {
            outputs.push(output_effect);
        }

        outputs
    }

    fn all_output_port_kinds(&self) -> OutputPortKinds {
        let mut outputs =
            TinyVec::with_capacity(self.outputs_len() + self.output_effect().is_some() as usize);

        outputs.extend(self.output_ports().map(|output| (output, EdgeKind::Value)));
        if let Some(output_effect) = self.output_effect() {
            outputs.push((output_effect, EdgeKind::Effect));
        }

        outputs
    }

    fn update_output(&mut self, from: OutputPort, to: OutputPort) {
        let parent_node = self.node;

        // Try to replace the output effect
        if let Some(output_effect) = self
            .output_effect_mut()
            .filter(|&&mut effect| effect == from)
        {
            tracing::trace!(
                node = ?parent_node,
                "replaced output effect {:?} of Theta with {:?}",
                from, to,
            );

            *output_effect = to;

        // Try to replace the output ports
        } else if let Some((output, node_id)) = {
            let mut output_node = None;
            self.outputs.retain(|&output, &mut node| {
                if output == from {
                    debug_assert!(output_node.is_none());
                    output_node = Some((output, node));

                    false
                } else {
                    true
                }
            });

            output_node
        } {
            tracing::trace!(
                node = ?parent_node,
                "replaced variant output {:?} of Theta with {:?}",
                from, to,
            );

            self.outputs.insert(to, node_id).debug_unwrap_none();

            let input = self.output_feedback.remove(&output).unwrap();
            self.output_feedback.insert(to, input).debug_unwrap_none();

        // Otherwise the theta doesn't have this output port
        } else {
            tracing::trace!(
                node = ?parent_node,
                "tried to replace output effect {:?} of Theta with {:?} but Theta doesn't have that port",
                from, to,
            );
        }
    }
}

/// The effects of a theta node
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::graph) struct ThetaEffects {
    /// The input effect's port on the theta node
    input: InputPort,
    /// The output effect's port on the theta node
    output: OutputPort,
}

impl ThetaEffects {
    pub const fn new(input: InputPort, output: OutputPort) -> Self {
        Self { input, output }
    }

    #[allow(dead_code)]
    pub const fn output(&self) -> OutputPort {
        self.output
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThetaData {
    pub(in crate::graph) outputs: Box<[OutputPort]>,
    pub(in crate::graph) condition: OutputPort,
    pub(in crate::graph) effect: OutputPort,
}

impl ThetaData {
    pub fn new<O>(outputs: O, condition: OutputPort, effect: OutputPort) -> Self
    where
        O: IntoIterator<Item = OutputPort>,
    {
        Self {
            outputs: outputs.into_iter().collect(),
            condition,
            effect,
        }
    }
}
