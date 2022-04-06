use crate::graph::{
    nodes::node_ext::{InputPortKinds, InputPorts, OutputPortKinds, OutputPorts},
    EdgeDescriptor, EdgeKind, InputPort, Node, NodeExt, NodeId, OutputPort, Rvsdg,
};
use tinyvec::tiny_vec;

// TODO: Replace Eq and Neq with Cmp node generic over all comparison types

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eq {
    node: NodeId,
    lhs: InputPort,
    rhs: InputPort,
    value: OutputPort,
}

impl Eq {
    pub(in crate::graph) const fn new(
        node: NodeId,
        lhs: InputPort,
        rhs: InputPort,
        value: OutputPort,
    ) -> Self {
        Self {
            node,
            lhs,
            rhs,
            value,
        }
    }

    /// Get the eq's left hand side
    pub const fn lhs(&self) -> InputPort {
        self.lhs
    }

    /// Get the eq's right hand side
    pub const fn rhs(&self) -> InputPort {
        self.rhs
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }
}

impl NodeExt for Eq {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::from_values(2)
    }

    fn all_input_ports(&self) -> InputPorts {
        tiny_vec![self.lhs, self.rhs]
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        tiny_vec! {
            [_; 4] =>
                (self.lhs, EdgeKind::Value),
                (self.rhs, EdgeKind::Value),
        }
    }

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        if self.lhs == from {
            self.lhs = to;
        }

        if self.rhs == from {
            self.rhs = to;
        }
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
pub struct Neq {
    node: NodeId,
    lhs: InputPort,
    rhs: InputPort,
    value: OutputPort,
}

impl Neq {
    pub(in crate::graph) const fn new(
        node: NodeId,
        lhs: InputPort,
        rhs: InputPort,
        value: OutputPort,
    ) -> Self {
        Self {
            node,
            lhs,
            rhs,
            value,
        }
    }

    /// Get the not-eq's left hand side
    pub const fn lhs(&self) -> InputPort {
        self.lhs
    }

    /// Get the not-eq's right hand side
    pub const fn rhs(&self) -> InputPort {
        self.rhs
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }
}

impl NodeExt for Neq {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::from_values(2)
    }

    fn all_input_ports(&self) -> InputPorts {
        tiny_vec![self.lhs, self.rhs]
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        tiny_vec! {
            [_; 4] =>
                (self.lhs, EdgeKind::Value),
                (self.rhs, EdgeKind::Value),
        }
    }

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        if self.lhs == from {
            self.lhs = to;
        }

        if self.rhs == from {
            self.rhs = to;
        }
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
pub enum EqOrNeq {
    Eq(Eq),
    Neq(Neq),
}

impl EqOrNeq {
    pub const fn lhs(&self) -> InputPort {
        match self {
            Self::Eq(eq) => eq.lhs(),
            Self::Neq(neq) => neq.lhs(),
        }
    }

    pub const fn rhs(&self) -> InputPort {
        match self {
            Self::Eq(eq) => eq.rhs(),
            Self::Neq(neq) => neq.rhs(),
        }
    }

    pub const fn value(&self) -> OutputPort {
        match self {
            Self::Eq(eq) => eq.value(),
            Self::Neq(neq) => neq.value(),
        }
    }

    pub fn cast_output_dest(graph: &Rvsdg, output: OutputPort) -> Option<Self> {
        graph.output_dest_node(output)?.try_into().ok()
    }

    pub fn cast_input_source(graph: &Rvsdg, input: InputPort) -> Option<Self> {
        graph.input_source_node(input).try_into().ok()
    }
}

impl TryFrom<Node> for EqOrNeq {
    type Error = Node;

    fn try_from(node: Node) -> Result<Self, Self::Error> {
        match node {
            Node::Eq(eq) => Ok(Self::Eq(eq)),
            Node::Neq(neq) => Ok(Self::Neq(neq)),
            node => Err(node),
        }
    }
}

impl<'a> TryFrom<&'a Node> for EqOrNeq {
    type Error = &'a Node;

    fn try_from(node: &'a Node) -> Result<Self, Self::Error> {
        match node {
            &Node::Eq(eq) => Ok(Self::Eq(eq)),
            &Node::Neq(neq) => Ok(Self::Neq(neq)),
            node => Err(node),
        }
    }
}
