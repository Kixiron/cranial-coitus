use crate::graph::{
    nodes::node_ext::{InputPortKinds, InputPorts, OutputPortKinds, OutputPorts},
    EdgeDescriptor, EdgeKind, InputPort, NodeExt, NodeId, OutputPort,
};
use tinyvec::{tiny_vec, TinyVec};

// TODO: Byte node

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Int {
    node: NodeId,
    value: OutputPort,
}

impl Int {
    pub(in crate::graph) const fn new(node: NodeId, value: OutputPort) -> Self {
        Self { node, value }
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }
}

impl NodeExt for Int {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::zero()
    }

    fn all_input_ports(&self) -> InputPorts {
        TinyVec::new()
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        TinyVec::new()
    }

    fn update_input(&mut self, _from: InputPort, _to: InputPort) {
        // TODO: Should this be a panic or warning?
    }

    fn output_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::from_values(1)
    }

    fn all_output_ports(&self) -> OutputPorts {
        tiny_vec![self.value]
    }

    fn all_output_port_kinds(&self) -> OutputPortKinds {
        tiny_vec! {
            [_; 4] => (self.value, EdgeKind::Value),
        }
    }

    fn update_output(&mut self, from: OutputPort, to: OutputPort) {
        if self.value == from {
            self.value = to;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bool {
    node: NodeId,
    value: OutputPort,
}

impl Bool {
    pub(in crate::graph) const fn new(node: NodeId, value: OutputPort) -> Self {
        Self { node, value }
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }
}

impl NodeExt for Bool {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::zero()
    }

    fn all_input_ports(&self) -> InputPorts {
        TinyVec::new()
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        TinyVec::new()
    }

    fn update_input(&mut self, _from: InputPort, _to: InputPort) {
        // TODO: Should this be a panic or warning?
    }

    fn output_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::from_values(1)
    }

    fn all_output_ports(&self) -> OutputPorts {
        tiny_vec![self.value]
    }

    fn all_output_port_kinds(&self) -> OutputPortKinds {
        tiny_vec! {
            [_; 4] => (self.value, EdgeKind::Value),
        }
    }

    fn update_output(&mut self, from: OutputPort, to: OutputPort) {
        if self.value == from {
            self.value = to;
        }
    }
}
