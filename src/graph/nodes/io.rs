//! IO related functions

use crate::graph::{
    nodes::node_ext::{InputPortKinds, InputPorts, OutputPortKinds, OutputPorts},
    EdgeCount, EdgeDescriptor, EdgeKind, InputPort, NodeExt, NodeId, OutputPort,
};
use tinyvec::{tiny_vec, TinyVec};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Input {
    node: NodeId,
    input_effect: InputPort,
    output_value: OutputPort,
    output_effect: OutputPort,
}

impl Input {
    pub(in crate::graph) const fn new(
        node: NodeId,
        input_effect: InputPort,
        output_value: OutputPort,
        output_effect: OutputPort,
    ) -> Self {
        Self {
            node,
            input_effect,
            output_value,
            output_effect,
        }
    }

    pub const fn input_effect(&self) -> InputPort {
        self.input_effect
    }

    pub const fn output_value(&self) -> OutputPort {
        self.output_value
    }

    pub const fn output_effect(&self) -> OutputPort {
        self.output_effect
    }
}

impl NodeExt for Input {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::from_effects(1)
    }

    fn all_input_ports(&self) -> InputPorts {
        tiny_vec![self.input_effect]
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        tiny_vec! {
            [_; 4] => (self.input_effect, EdgeKind::Effect),
        }
    }

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        if from == self.input_effect {
            self.input_effect = to;
        }
    }

    fn output_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::new(EdgeCount::one(), EdgeCount::one())
    }

    fn all_output_ports(&self) -> OutputPorts {
        tiny_vec![self.output_value, self.output_effect]
    }

    fn all_output_port_kinds(&self) -> OutputPortKinds {
        tiny_vec! {
            [_; 4] =>
                (self.output_value, EdgeKind::Value),
                (self.output_effect, EdgeKind::Effect),
        }
    }

    fn update_output(&mut self, from: OutputPort, to: OutputPort) {
        if self.output_value == from {
            self.output_value = to;
        }

        if self.output_effect == from {
            self.output_effect = to;
        }
    }
}

pub type OutputValues = TinyVec<[InputPort; 1]>;

#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    node: NodeId,
    values: OutputValues,
    input_effect: InputPort,
    output_effect: OutputPort,
}

impl Output {
    pub(in crate::graph) const fn new(
        node: NodeId,
        values: OutputValues,
        input_effect: InputPort,
        output_effect: OutputPort,
    ) -> Self {
        Self {
            node,
            values,
            input_effect,
            output_effect,
        }
    }

    pub fn values(&self) -> &[InputPort] {
        &*self.values
    }

    pub fn values_mut(&mut self) -> &mut OutputValues {
        &mut self.values
    }

    pub const fn input_effect(&self) -> InputPort {
        self.input_effect
    }

    pub const fn output_effect(&self) -> OutputPort {
        self.output_effect
    }
}

impl NodeExt for Output {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::new(EdgeCount::one(), EdgeCount::new(Some(1), None))
    }

    fn all_input_ports(&self) -> InputPorts {
        let mut ports = TinyVec::with_capacity(self.values.len() + 1);
        ports.extend_from_slice(&self.values);
        ports.push(self.input_effect);
        ports
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        let mut ports = TinyVec::with_capacity(self.values.len() + 1);
        ports.extend(self.values.iter().map(|&value| (value, EdgeKind::Value)));
        ports.push((self.input_effect, EdgeKind::Effect));
        ports
    }

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        for value in &mut self.values {
            if from == *value {
                *value = to;
            }
        }

        if from == self.input_effect {
            self.input_effect = to;
        }
    }

    fn output_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::from_effects(1)
    }

    fn all_output_ports(&self) -> OutputPorts {
        tiny_vec![self.output_effect]
    }

    fn all_output_port_kinds(&self) -> OutputPortKinds {
        tiny_vec! {
            [_; 4] => (self.output_effect, EdgeKind::Effect),
        }
    }

    fn update_output(&mut self, from: OutputPort, to: OutputPort) {
        if self.output_effect == from {
            self.output_effect = to;
        }
    }
}
