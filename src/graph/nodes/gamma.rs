//! The gamma node

use crate::graph::{
    nodes::node_ext::{InputPortKinds, OutputPortKinds},
    EdgeCount, EdgeDescriptor, EdgeKind, InputParam, InputPort, NodeExt, NodeId, OutputPort, Rvsdg,
};
use tinyvec::TinyVec;

// TODO: Refactor this
#[derive(Debug, Clone, PartialEq)]
pub struct Gamma {
    node: NodeId,
    inputs: TinyVec<[InputPort; 4]>,
    // TODO: Make optional
    effect_in: InputPort,
    input_params: TinyVec<[[NodeId; 2]; 4]>,
    input_effect: OutputPort,
    outputs: TinyVec<[OutputPort; 4]>,
    effect_out: OutputPort,
    output_params: TinyVec<[[NodeId; 2]; 4]>,
    // TODO: Make optional, linked with `effect_in`
    output_effects: [OutputPort; 2],
    start_nodes: [NodeId; 2],
    end_nodes: [NodeId; 2],
    bodies: Box<[Rvsdg; 2]>,
    condition: InputPort,
}

impl Gamma {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::graph) fn new(
        node: NodeId,
        inputs: TinyVec<[InputPort; 4]>,
        effect_in: InputPort,
        input_params: TinyVec<[[NodeId; 2]; 4]>,
        input_effect: OutputPort,
        outputs: TinyVec<[OutputPort; 4]>,
        effect_out: OutputPort,
        output_params: TinyVec<[[NodeId; 2]; 4]>,
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

    pub const fn starts(&self) -> [NodeId; 2] {
        self.start_nodes
    }

    pub const fn ends(&self) -> [NodeId; 2] {
        self.end_nodes
    }

    pub fn inputs(&self) -> &[InputPort] {
        &self.inputs
    }

    pub fn inputs_mut(&mut self) -> &mut TinyVec<[InputPort; 4]> {
        &mut self.inputs
    }

    pub const fn condition(&self) -> InputPort {
        self.condition
    }

    pub const fn input_effect(&self) -> InputPort {
        self.effect_in
    }

    pub const fn output_effect(&self) -> OutputPort {
        self.effect_out
    }

    pub fn input_params(&self) -> &[[NodeId; 2]] {
        &self.input_params
    }

    pub fn input_params_mut(&mut self) -> &mut TinyVec<[[NodeId; 2]; 4]> {
        &mut self.input_params
    }

    pub fn outputs(&self) -> &[OutputPort] {
        &self.outputs
    }

    #[allow(dead_code)]
    pub fn outputs_mut(&mut self) -> &mut TinyVec<[OutputPort; 4]> {
        &mut self.outputs
    }

    pub fn output_params(&self) -> &[[NodeId; 2]] {
        &self.output_params
    }

    pub fn output_params_mut(&mut self) -> &mut TinyVec<[[NodeId; 2]; 4]> {
        &mut self.output_params
    }

    pub const fn true_branch(&self) -> &Rvsdg {
        &self.bodies[0]
    }

    pub fn true_mut(&mut self) -> &mut Rvsdg {
        &mut self.bodies[0]
    }

    pub const fn false_branch(&self) -> &Rvsdg {
        &self.bodies[1]
    }

    pub fn false_mut(&mut self) -> &mut Rvsdg {
        &mut self.bodies[1]
    }

    pub fn add_param(
        &mut self,
        graph: &mut Rvsdg,
        source: OutputPort,
    ) -> (InputPort, [InputParam; 2]) {
        let true_param = self.bodies[0].input_param(EdgeKind::Value);
        let false_param = self.bodies[1].input_param(EdgeKind::Value);
        self.input_params
            .push([true_param.node(), false_param.node()]);

        let input = graph.create_input_port(self.node, EdgeKind::Value);
        graph.add_value_edge(source, input);

        (input, [true_param, false_param])
    }

    pub fn paired_inputs(&self) -> impl Iterator<Item = (InputPort, [NodeId; 2])> + '_ {
        self.inputs()
            .iter()
            .copied()
            .zip(self.input_params().iter().copied())
    }

    pub fn true_input_pairs(&self) -> impl Iterator<Item = (InputPort, NodeId)> + '_ {
        self.paired_inputs()
            .map(|(input, [true_param, _])| (input, true_param))
    }

    pub fn false_input_pairs(&self) -> impl Iterator<Item = (InputPort, NodeId)> + '_ {
        self.paired_inputs()
            .map(|(input, [_, false_param])| (input, false_param))
    }

    pub fn paired_outputs(&self) -> impl Iterator<Item = (OutputPort, [NodeId; 2])> + '_ {
        self.outputs()
            .iter()
            .copied()
            .zip(self.output_params().iter().copied())
    }

    pub fn true_output_pairs(&self) -> impl Iterator<Item = (OutputPort, NodeId)> + '_ {
        self.paired_outputs()
            .map(|(output, [true_param, _])| (output, true_param))
    }

    pub fn false_output_pairs(&self) -> impl Iterator<Item = (OutputPort, NodeId)> + '_ {
        self.paired_outputs()
            .map(|(output, [_, false_param])| (output, false_param))
    }
}

impl NodeExt for Gamma {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::new(EdgeCount::one(), EdgeCount::new(Some(2), None))
    }

    fn all_input_ports(&self) -> TinyVec<[InputPort; 4]> {
        let mut inputs = TinyVec::with_capacity(self.inputs.len() + 2);
        inputs.push(self.condition);
        inputs.extend(self.inputs.iter().copied());
        inputs.push(self.effect_in);
        inputs
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        let mut inputs = TinyVec::with_capacity(self.inputs.len() + 2);
        inputs.push((self.condition, EdgeKind::Value));
        inputs.extend(
            self.inputs
                .iter()
                .copied()
                .map(|input| (input, EdgeKind::Value)),
        );
        inputs.push((self.effect_in, EdgeKind::Effect));
        inputs
    }

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        if self.condition == from {
            self.condition = to;
        }

        for input in &mut self.inputs {
            if *input == from {
                *input = to;
            }
        }

        if self.effect_in == from {
            self.effect_in = to;
        }
    }

    fn output_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::new(EdgeCount::one(), EdgeCount::unlimited())
    }

    fn all_output_ports(&self) -> TinyVec<[OutputPort; 4]> {
        let mut outputs = TinyVec::with_capacity(self.outputs.len() + 1);
        outputs.extend(self.outputs.iter().copied());
        outputs.push(self.effect_out);
        outputs
    }

    fn all_output_port_kinds(&self) -> OutputPortKinds {
        let mut outputs = TinyVec::with_capacity(self.outputs.len() + 1);
        outputs.extend(
            self.outputs
                .iter()
                .copied()
                .map(|output| (output, EdgeKind::Value)),
        );
        outputs.push((self.effect_out, EdgeKind::Effect));
        outputs
    }

    fn update_output(&mut self, from: OutputPort, to: OutputPort) {
        for output in &mut self.outputs {
            if *output == from {
                *output = to;
            }
        }

        if self.effect_out == from {
            self.effect_out = to;
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GammaData {
    pub(in crate::graph) outputs: Box<[OutputPort]>,
    pub(in crate::graph) effect: OutputPort,
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
