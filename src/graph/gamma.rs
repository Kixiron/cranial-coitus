use crate::graph::{InputPort, NodeId, OutputPort, Rvsdg};

#[derive(Debug, Clone, PartialEq)]
pub struct Gamma {
    pub(super) node: NodeId,
    pub(super) inputs: Vec<InputPort>,
    // TODO: Make optional
    pub(super) effect_in: InputPort,
    input_params: Vec<[NodeId; 2]>,
    input_effect: OutputPort,
    pub(super) outputs: Vec<OutputPort>,
    pub(super) effect_out: OutputPort,
    output_params: Vec<[NodeId; 2]>,
    // TODO: Make optional, linked with `effect_in`
    output_effects: [OutputPort; 2],
    start_nodes: [NodeId; 2],
    end_nodes: [NodeId; 2],
    bodies: Box<[Rvsdg; 2]>,
    pub(super) condition: InputPort,
}

impl Gamma {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node: NodeId,
        inputs: Vec<InputPort>,
        effect_in: InputPort,
        input_params: Vec<[NodeId; 2]>,
        input_effect: OutputPort,
        outputs: Vec<OutputPort>,
        effect_out: OutputPort,
        output_params: Vec<[NodeId; 2]>,
        output_effects: [OutputPort; 2],
        start_nodes: [NodeId; 2],
        end_nodes: [NodeId; 2],
        bodies: Box<[Rvsdg; 2]>,
        condition: InputPort,
    ) -> Self {
        Self {
            node,
            inputs,
            effect_in,
            input_params,
            input_effect,
            outputs,
            effect_out,
            output_params,
            output_effects,
            start_nodes,
            end_nodes,
            bodies,
            condition,
        }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn starts(&self) -> [NodeId; 2] {
        self.start_nodes
    }

    pub const fn ends(&self) -> [NodeId; 2] {
        self.end_nodes
    }

    pub fn inputs(&self) -> &[InputPort] {
        &self.inputs
    }

    pub(super) fn inputs_mut(&mut self) -> &mut Vec<InputPort> {
        &mut self.inputs
    }

    pub const fn condition(&self) -> InputPort {
        self.condition
    }

    pub const fn effect_in(&self) -> InputPort {
        self.effect_in
    }

    pub const fn effect_out(&self) -> OutputPort {
        self.effect_out
    }

    pub fn input_params(&self) -> &[[NodeId; 2]] {
        &self.input_params
    }

    pub fn outputs(&self) -> &[OutputPort] {
        &self.outputs
    }

    #[allow(dead_code)]
    pub fn outputs_mut(&mut self) -> &mut Vec<OutputPort> {
        &mut self.outputs
    }

    pub fn output_params(&self) -> &[[NodeId; 2]] {
        &self.output_params
    }

    pub const fn true_branch(&self) -> &Rvsdg {
        &self.bodies[0]
    }

    pub fn truthy_mut(&mut self) -> &mut Rvsdg {
        &mut self.bodies[0]
    }

    pub const fn false_branch(&self) -> &Rvsdg {
        &self.bodies[1]
    }

    pub fn falsy_mut(&mut self) -> &mut Rvsdg {
        &mut self.bodies[1]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GammaData {
    pub(super) outputs: Box<[OutputPort]>,
    pub(super) effect: OutputPort,
}

impl GammaData {
    pub fn new<O>(outputs: O, effect: OutputPort) -> Self
    where
        O: IntoIterator<Item = OutputPort>,
    {
        Self {
            outputs: outputs.into_iter().collect(),
            effect,
        }
    }
}
