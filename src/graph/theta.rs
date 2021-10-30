use crate::graph::{
    EdgeCount, EdgeDescriptor, End, InputParam, InputPort, NodeExt, NodeId, OutputParam,
    OutputPort, Rvsdg, Start, Subgraph,
};
use std::collections::BTreeMap;
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
    invariant_inputs: BTreeMap<InputPort, NodeId>,

    /// These inputs change upon each iteration, they're connected to `output_back_edges`
    ///
    /// These point from an [`InputPort`] on the theta to the [`NodeId`]
    /// of an [`InputParam`] within the theta's body
    variant_inputs: BTreeMap<InputPort, NodeId>,

    /// The node's outputs, these all must be included in `output_back_edges`. Any invariant
    /// data must go around the loop to reach dependents after it
    ///
    /// These point from an [`OutputPort`] on the theta to the [`NodeId`]
    /// of an [`OutputParam`] within the theta's body
    outputs: BTreeMap<OutputPort, NodeId>,

    /// These are the relationships between outputs and variant inputs
    ///
    /// These point from an [`OutputPort`] in `outputs` to an [`InputPort`] in `variant_inputs`
    output_feedback: BTreeMap<OutputPort, InputPort>,

    /// The theta's condition, it should be an expression that evaluates to a boolean
    ///
    /// Points to an [`OutputParam`] within the theta's body
    condition: NodeId,

    /// The theta's loop body, contains the [`Start`] and [`End`] [`NodeId`]s
    ///
    /// [`Start`]: crate::graph::Start
    /// [`End`]: crate::graph::End
    subgraph: Box<Subgraph>,
}

impl Theta {
    /// Create a new theta node
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node: NodeId,
        effects: Option<ThetaEffects>,
        invariant_inputs: BTreeMap<InputPort, NodeId>,
        variant_inputs: BTreeMap<InputPort, NodeId>,
        outputs: BTreeMap<OutputPort, NodeId>,
        output_feedback: BTreeMap<OutputPort, InputPort>,
        condition: NodeId,
        subgraph: Box<Subgraph>,
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
    pub fn condition(&self) -> OutputParam {
        self.subgraph.to_node(self.condition)
    }

    /// Get the [`Start`] of the theta's body
    pub fn start_node(&self) -> Start {
        self.subgraph.start_node()
    }

    /// Get the [`End`] of the theta's body
    pub fn end_node(&self) -> End {
        self.subgraph.end_node()
    }

    /// Get the [`NodeId`] of the theta body's [`End`] node
    pub const fn end_node_id(&self) -> NodeId {
        self.subgraph.end
    }

    /// Get access to the theta node's body
    pub fn body(&self) -> &Rvsdg {
        &self.subgraph
    }

    /// Get mutable access to the theta node's body
    pub fn body_mut(&mut self) -> &mut Rvsdg {
        &mut self.subgraph
    }

    /// Get the input effect's port from the theta node if available
    pub fn input_effect(&self) -> Option<InputPort> {
        self.effects.map(|effects| effects.input)
    }

    /// Get the output effect's port from the theta node if available
    pub fn output_effect(&self) -> Option<OutputPort> {
        self.effects.map(|effects| effects.output)
    }

    pub fn effects(&self) -> Option<(InputPort, OutputPort)> {
        self.effects.map(|effects| (effects.input, effects.output))
    }

    pub const fn has_effects(&self) -> bool {
        self.effects.is_some()
    }

    /// Get the theta's output feedback edges
    pub const fn output_feedback(&self) -> &BTreeMap<OutputPort, InputPort> {
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
}

/// Input/Output port related functions
impl Theta {
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
    pub fn input_pairs(&self) -> impl Iterator<Item = (InputPort, InputParam)> + '_ {
        self.invariant_input_pairs()
            .chain(self.variant_input_pairs())
    }

    /// Returns the input ports and the associated node id of each input param for all inputs (variant and invariant)
    /// to the theta node, *not including effect inputs*
    pub fn input_pair_ids(&self) -> impl Iterator<Item = (InputPort, NodeId)> + '_ {
        self.invariant_input_pair_ids()
            .chain(self.variant_input_pair_ids())
    }

    /// Returns all input params to the theta node, *not including effect inputs*
    pub fn input_params(&self) -> impl Iterator<Item = InputParam> + '_ {
        self.invariant_input_params()
            .chain(self.variant_input_params())
    }

    /// Returns the node ids of all input params to the theta node (variant and invariant),
    /// *not including effect inputs*
    pub fn input_param_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.invariant_input_param_ids()
            .chain(self.variant_input_param_ids())
    }

    pub fn input_pair_ids_with_feedback(
        &self,
    ) -> impl Iterator<Item = (InputPort, NodeId, OutputPort)> + '_ {
        self.invariant_input_pair_ids_with_feedback()
            .chain(self.variant_input_pair_ids_with_feedback())
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
    pub fn invariant_input_pairs(&self) -> impl Iterator<Item = (InputPort, InputParam)> + '_ {
        self.invariant_inputs
            .iter()
            .map(|(&port, &param)| (port, self.subgraph.to_node(param)))
    }

    /// Returns the node ids of each invariant input and the input port that feeds into them
    pub fn invariant_input_pair_ids(&self) -> impl Iterator<Item = (InputPort, NodeId)> + '_ {
        self.invariant_inputs
            .iter()
            .map(|(&port, &param)| (port, param))
    }

    /// Returns the invariant inputs to the theta node
    pub fn invariant_input_params(&self) -> impl Iterator<Item = InputParam> + '_ {
        self.invariant_inputs
            .iter()
            .map(|(_, &param)| self.subgraph.to_node(param))
    }

    /// Returns the node ids of invariant inputs to the theta node
    pub fn invariant_input_param_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.invariant_inputs.values().copied()
    }

    pub fn invariant_input_pair_ids_with_feedback(
        &self,
    ) -> impl Iterator<Item = (InputPort, NodeId, OutputPort)> + '_ {
        self.invariant_inputs.iter().map(|(&port, &param)| {
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

    pub fn replace_invariant_inputs(&mut self, invariant_inputs: BTreeMap<InputPort, NodeId>) {
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
    pub fn variant_input_pairs(&self) -> impl Iterator<Item = (InputPort, InputParam)> + '_ {
        self.variant_inputs
            .iter()
            .map(|(&port, &param)| (port, self.subgraph.to_node(param)))
    }

    /// Returns the node ids of each variant input and the input port that feeds into them
    pub fn variant_input_pair_ids(&self) -> impl Iterator<Item = (InputPort, NodeId)> + '_ {
        self.variant_inputs
            .iter()
            .map(|(&port, &param)| (port, param))
    }

    /// Returns the variant inputs to the theta node
    pub fn variant_input_params(&self) -> impl Iterator<Item = InputParam> + '_ {
        self.variant_inputs
            .iter()
            .map(|(_, &param)| self.subgraph.to_node(param))
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

    pub fn replace_variant_inputs(&mut self, variant_inputs: BTreeMap<InputPort, NodeId>) {
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
    pub fn output_pairs(&self) -> impl Iterator<Item = (OutputPort, OutputParam)> + '_ {
        self.outputs
            .iter()
            .map(|(&port, &param)| (port, self.subgraph.to_node(param)))
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

    pub fn replace_outputs(&mut self, outputs: BTreeMap<OutputPort, NodeId>) {
        self.outputs = outputs;
    }

    pub fn replace_output_feedback(&mut self, output_feedback: BTreeMap<OutputPort, InputPort>) {
        self.output_feedback = output_feedback;
    }

    /// Returns the outputs from the theta node, *not including effect outputs*
    pub fn output_params(&self) -> impl Iterator<Item = OutputParam> + '_ {
        self.outputs
            .iter()
            .map(|(_, &param)| self.subgraph.to_node(param))
    }

    /// Returns the node ids of outputs from the theta node, *not including effect outputs*
    pub fn output_param_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.outputs.values().copied()
    }

    /// Returns all variant inputs to the theta node along with the output
    /// that loops back into the given input
    pub fn variant_inputs_loopback(&self) -> impl Iterator<Item = (InputParam, OutputParam)> + '_ {
        self.output_feedback.iter().map(|(output, input)| {
            let (input, output) = (self.variant_inputs[input], self.outputs[output]);
            (self.subgraph.to_node(input), self.subgraph.to_node(output))
        })
    }
}

/// Utility functions
// TODO: Function to inline the theta's body into the given graph
impl Theta {
    pub fn get_output_param(&self, port: OutputPort) -> Option<OutputParam> {
        self.outputs
            .get(&port)
            .map(|&node| self.subgraph.to_node(node))
    }

    /// Returns `true` if the theta's condition is always `false`
    pub fn is_infinite(&self) -> bool {
        let cond = self.condition();
        let condition_is_false = self
            .body()
            .input_source_node(cond.input())
            .as_bool()
            .map_or(false, |(_, value)| value);

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

    fn all_input_ports(&self) -> TinyVec<[InputPort; 4]> {
        let mut inputs =
            TinyVec::with_capacity(self.inputs_len() + self.input_effect().is_some() as usize);

        inputs.extend(self.input_ports());
        if let Some(input_effect) = self.input_effect() {
            inputs.push(input_effect);
        }

        inputs
    }

    fn output_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::new(
            EdgeCount::exact(self.output_effect().is_some() as usize),
            EdgeCount::new(None, Some(self.outputs_len())),
        )
    }

    fn all_output_ports(&self) -> TinyVec<[OutputPort; 4]> {
        let mut outputs =
            TinyVec::with_capacity(self.outputs_len() + self.output_effect().is_some() as usize);

        outputs.extend(self.output_ports());
        if let Some(output_effect) = self.output_effect() {
            outputs.push(output_effect);
        }

        outputs
    }
}

/// The effects of a theta node
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThetaEffects {
    /// The input effect's port on the theta node
    input: InputPort,
    /// The output effect's port on the theta node
    pub(super) output: OutputPort,
}

impl ThetaEffects {
    pub(super) const fn new(input: InputPort, output: OutputPort) -> Self {
        Self { input, output }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThetaData {
    pub(super) outputs: Box<[OutputPort]>,
    pub(super) condition: OutputPort,
    pub(super) effect: OutputPort,
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

#[derive(Debug, Clone, PartialEq)]
pub struct ThetaStub {
    output_effect: Option<OutputPort>,
    outputs: TinyVec<[OutputPort; 5]>,
}

impl ThetaStub {
    pub(super) const fn new(
        output_effect: Option<OutputPort>,
        outputs: TinyVec<[OutputPort; 5]>,
    ) -> Self {
        Self {
            output_effect,
            outputs,
        }
    }

    /// Get the output effect's port from the theta node if available
    pub fn output_effect(&self) -> Option<OutputPort> {
        self.output_effect
    }

    pub fn outputs(&self) -> &[OutputPort] {
        &self.outputs
    }
}
