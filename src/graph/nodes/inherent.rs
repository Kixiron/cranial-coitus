//! Nodes inherent to the structure of the graph

use crate::graph::{
    nodes::node_ext::{InputPortKinds, OutputPortKinds},
    EdgeCount, EdgeDescriptor, EdgeKind, InputPort, NodeExt, NodeId, OutputPort,
};
use tinyvec::{tiny_vec, TinyVec};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Start {
    node: NodeId,
    effect: OutputPort,
}

impl Start {
    pub(in crate::graph) const fn new(node: NodeId, effect: OutputPort) -> Self {
        Self { node, effect }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn effect(&self) -> OutputPort {
        self.effect
    }
}

impl NodeExt for Start {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::zero()
    }

    fn all_input_ports(&self) -> TinyVec<[InputPort; 4]> {
        TinyVec::new()
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        TinyVec::new()
    }

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        tracing::trace!(
            node = ?self.node,
            "tried to replace input port {:?} of Start with {:?} but Start doesn't have any inputs",
            from, to,
        );
    }

    fn output_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero())
    }

    fn all_output_ports(&self) -> TinyVec<[OutputPort; 4]> {
        tiny_vec![self.effect]
    }

    fn all_output_port_kinds(&self) -> OutputPortKinds {
        tiny_vec! {
            [_; 4] => (self.effect, EdgeKind::Effect),
        }
    }

    fn update_output(&mut self, from: OutputPort, to: OutputPort) {
        if self.effect == from {
            self.effect = to;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct End {
    node: NodeId,
    input_effect: InputPort,
}

impl End {
    pub(in crate::graph) const fn new(node: NodeId, input_effect: InputPort) -> Self {
        Self { node, input_effect }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn input_effect(&self) -> InputPort {
        self.input_effect
    }
}

impl NodeExt for End {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero())
    }

    fn all_input_ports(&self) -> TinyVec<[InputPort; 4]> {
        tiny_vec![self.input_effect]
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        tiny_vec! {
            [_; 4] => (self.input_effect, EdgeKind::Effect),
        }
    }

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        if self.input_effect == from {
            tracing::trace!(
                node = ?self.node,
                "replaced input effect {:?} of End with {:?}",
                from, to,
            );

            self.input_effect = to;
        } else {
            tracing::trace!(
                node = ?self.node,
                "tried to replace input effect {:?} of End with {:?} but End doesn't have that port",
                from, to,
            );
        }
    }

    fn output_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::zero()
    }

    fn all_output_ports(&self) -> TinyVec<[OutputPort; 4]> {
        TinyVec::new()
    }

    fn all_output_port_kinds(&self) -> OutputPortKinds {
        TinyVec::new()
    }

    fn update_output(&mut self, _from: OutputPort, _to: OutputPort) {
        // TODO: Should this panic or warn?
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InputParam {
    node: NodeId,
    output: OutputPort,
    kind: EdgeKind,
}

impl InputParam {
    pub(in crate::graph) const fn new(node: NodeId, output: OutputPort, kind: EdgeKind) -> Self {
        Self { node, output, kind }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn output(&self) -> OutputPort {
        self.output
    }
}

impl NodeExt for InputParam {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::zero()
    }

    fn all_input_ports(&self) -> TinyVec<[InputPort; 4]> {
        TinyVec::new()
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        TinyVec::new()
    }

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        tracing::trace!(
            node = ?self.node,
            "tried to replace input port {:?} of InputParam with {:?} but InputParam doesn't have any inputs",
            from, to,
        );
    }

    fn output_desc(&self) -> EdgeDescriptor {
        match self.kind {
            EdgeKind::Effect => EdgeDescriptor::from_effects(1),
            EdgeKind::Value => EdgeDescriptor::from_values(1),
        }
    }

    fn all_output_ports(&self) -> TinyVec<[OutputPort; 4]> {
        tiny_vec![self.output]
    }

    fn all_output_port_kinds(&self) -> OutputPortKinds {
        tiny_vec! {
            [_; 4] => (self.output, self.kind),
        }
    }

    fn update_output(&mut self, from: OutputPort, to: OutputPort) {
        if self.output == from {
            self.output = to;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutputParam {
    node: NodeId,
    input: InputPort,
    kind: EdgeKind,
}

impl OutputParam {
    pub(in crate::graph) const fn new(node: NodeId, input: InputPort, kind: EdgeKind) -> Self {
        Self { node, input, kind }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn input(&self) -> InputPort {
        self.input
    }
}

impl NodeExt for OutputParam {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        match self.kind {
            EdgeKind::Effect => EdgeDescriptor::from_effects(1),
            EdgeKind::Value => EdgeDescriptor::from_values(1),
        }
    }

    fn all_input_ports(&self) -> TinyVec<[InputPort; 4]> {
        tiny_vec![self.input]
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        tiny_vec! {
            [_; 4] => (self.input, self.kind),
        }
    }

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        if self.input == from {
            tracing::trace!(
                node = ?self.node,
                "replaced input port {:?} of OutputParam with {:?}",
                from, to,
            );

            self.input = to;
        } else {
            tracing::trace!(
                node = ?self.node,
                "tried to replace input port {:?} of OutputParam with {:?} but OutputParam doesn't have that port",
                from, to,
            );
        }
    }

    fn output_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::zero()
    }

    fn all_output_ports(&self) -> TinyVec<[OutputPort; 4]> {
        TinyVec::new()
    }

    fn all_output_port_kinds(&self) -> OutputPortKinds {
        TinyVec::new()
    }

    fn update_output(&mut self, _from: OutputPort, _to: OutputPort) {
        // TODO: Should this panic or warn?
    }
}
