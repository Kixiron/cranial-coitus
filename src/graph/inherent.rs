use crate::graph::{EdgeCount, EdgeDescriptor, EdgeKind, InputPort, NodeExt, NodeId, OutputPort};
use tinyvec::{tiny_vec, TinyVec};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Start {
    node: NodeId,
    pub(super) effect: OutputPort,
}

impl Start {
    pub(super) const fn new(node: NodeId, effect: OutputPort) -> Self {
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct End {
    node: NodeId,
    pub(super) input_effect: InputPort,
}

impl End {
    pub(super) const fn new(node: NodeId, input_effect: InputPort) -> Self {
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InputParam {
    node: NodeId,
    pub(super) output: OutputPort,
    pub(super) kind: EdgeKind,
}

impl InputParam {
    pub(super) const fn new(node: NodeId, output: OutputPort, kind: EdgeKind) -> Self {
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

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        tracing::trace!(
            node = ?self.node,
            "tried to replace input port {:?} of InputParam with {:?} but InputParam doesn't have any inputs",
            from, to,
        );
    }

    fn output_desc(&self) -> EdgeDescriptor {
        match self.kind {
            EdgeKind::Effect => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
            EdgeKind::Value => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
        }
    }

    fn all_output_ports(&self) -> TinyVec<[OutputPort; 4]> {
        tiny_vec![self.output]
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutputParam {
    node: NodeId,
    pub(super) input: InputPort,
    pub(super) kind: EdgeKind,
}

impl OutputParam {
    pub(super) const fn new(node: NodeId, input: InputPort, kind: EdgeKind) -> Self {
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
            EdgeKind::Effect => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
            EdgeKind::Value => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
        }
    }

    fn all_input_ports(&self) -> TinyVec<[InputPort; 4]> {
        tiny_vec![self.input]
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
}
