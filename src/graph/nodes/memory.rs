//! Operations on program memory

use crate::graph::{
    nodes::node_ext::{InputPortKinds, InputPorts, OutputPortKinds, OutputPorts},
    EdgeCount, EdgeDescriptor, EdgeKind, InputPort, NodeExt, NodeId, OutputPort,
};
use tinyvec::tiny_vec;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Load {
    node: NodeId,
    ptr: InputPort,
    input_effect: InputPort,
    output_value: OutputPort,
    output_effect: OutputPort,
}

impl Load {
    pub(in crate::graph) const fn new(
        node: NodeId,
        ptr: InputPort,
        input_effect: InputPort,
        output_value: OutputPort,
        output_effect: OutputPort,
    ) -> Self {
        Self {
            node,
            ptr,
            input_effect,
            output_value,
            output_effect,
        }
    }

    pub const fn ptr(&self) -> InputPort {
        self.ptr
    }

    pub const fn output_value(&self) -> OutputPort {
        self.output_value
    }

    pub const fn input_effect(&self) -> InputPort {
        self.input_effect
    }

    pub const fn output_effect(&self) -> OutputPort {
        self.output_effect
    }
}

impl NodeExt for Load {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::new(EdgeCount::one(), EdgeCount::one())
    }

    fn all_input_ports(&self) -> InputPorts {
        tiny_vec![self.ptr, self.input_effect]
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        tiny_vec! {
            [_; 4] =>
                (self.ptr, EdgeKind::Value),
                (self.input_effect, EdgeKind::Effect),
        }
    }

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        if from == self.ptr {
            self.ptr = to;
        }

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Store {
    node: NodeId,
    ptr: InputPort,
    value: InputPort,
    input_effect: InputPort,
    output_effect: OutputPort,
}

impl Store {
    pub(in crate::graph) const fn new(
        node: NodeId,
        ptr: InputPort,
        value: InputPort,
        input_effect: InputPort,
        output_effect: OutputPort,
    ) -> Self {
        Self {
            node,
            ptr,
            value,
            input_effect,
            output_effect,
        }
    }

    pub const fn ptr(&self) -> InputPort {
        self.ptr
    }

    pub const fn value(&self) -> InputPort {
        self.value
    }

    pub const fn effect_in(&self) -> InputPort {
        self.input_effect
    }

    pub const fn output_effect(&self) -> OutputPort {
        self.output_effect
    }
}

impl NodeExt for Store {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::new(EdgeCount::one(), EdgeCount::two())
    }

    fn all_input_ports(&self) -> InputPorts {
        tiny_vec![self.ptr, self.value, self.input_effect]
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        tiny_vec! {
            [_; 4] =>
                (self.ptr, EdgeKind::Value),
                (self.value, EdgeKind::Value),
                (self.input_effect, EdgeKind::Effect),
        }
    }

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        if self.ptr == from {
            self.ptr = to;
        }

        if self.value == from {
            self.value = to;
        }

        if self.input_effect == from {
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
