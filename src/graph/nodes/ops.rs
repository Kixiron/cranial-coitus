//! Value operations

use crate::graph::{
    nodes::node_ext::{InputPortKinds, InputPorts, OutputPortKinds, OutputPorts},
    EdgeDescriptor, EdgeKind, InputPort, Node, NodeExt, NodeId, OutputPort, Rvsdg,
};
use tinyvec::tiny_vec;

// TODO: Div, Neq

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Add {
    node: NodeId,
    lhs: InputPort,
    rhs: InputPort,
    value: OutputPort,
}

impl Add {
    pub const fn new(node: NodeId, lhs: InputPort, rhs: InputPort, value: OutputPort) -> Self {
        Self {
            node,
            lhs,
            rhs,
            value,
        }
    }

    /// Get the add's left hand side
    pub const fn lhs(&self) -> InputPort {
        self.lhs
    }

    /// Get the add's right hand side
    pub const fn rhs(&self) -> InputPort {
        self.rhs
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }
}

impl NodeExt for Add {
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
pub struct Sub {
    node: NodeId,
    lhs: InputPort,
    rhs: InputPort,
    value: OutputPort,
}

impl Sub {
    pub const fn new(node: NodeId, lhs: InputPort, rhs: InputPort, value: OutputPort) -> Self {
        Self {
            node,
            lhs,
            rhs,
            value,
        }
    }

    /// Get the sub's left hand side
    pub const fn lhs(&self) -> InputPort {
        self.lhs
    }

    /// Get the sub's right hand side
    pub const fn rhs(&self) -> InputPort {
        self.rhs
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }
}

impl NodeExt for Sub {
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
pub struct Mul {
    node: NodeId,
    lhs: InputPort,
    rhs: InputPort,
    value: OutputPort,
}

impl Mul {
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

    /// Get the add's left hand side
    pub const fn lhs(&self) -> InputPort {
        self.lhs
    }

    /// Get the add's right hand side
    pub const fn rhs(&self) -> InputPort {
        self.rhs
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }
}

impl NodeExt for Mul {
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

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Not {
    node: NodeId,
    input: InputPort,
    value: OutputPort,
}

impl Not {
    pub(in crate::graph) const fn new(node: NodeId, input: InputPort, value: OutputPort) -> Self {
        Self { node, input, value }
    }

    /// Get the not's input
    pub const fn input(&self) -> InputPort {
        self.input
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }
}

impl NodeExt for Not {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::from_values(1)
    }

    fn all_input_ports(&self) -> InputPorts {
        tiny_vec![self.input]
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        tiny_vec! {
            [_; 4] => (self.input, EdgeKind::Value),
        }
    }

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        if self.input == from {
            self.input = to;
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
pub struct Neg {
    node: NodeId,
    input: InputPort,
    value: OutputPort,
}

impl Neg {
    pub(in crate::graph) const fn new(node: NodeId, input: InputPort, value: OutputPort) -> Self {
        Self { node, input, value }
    }

    pub const fn input(&self) -> InputPort {
        self.input
    }

    pub const fn value(&self) -> OutputPort {
        self.value
    }
}

impl NodeExt for Neg {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::from_values(1)
    }

    fn all_input_ports(&self) -> InputPorts {
        tiny_vec![self.input]
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        tiny_vec! {
            [_; 4] => (self.input, EdgeKind::Value),
        }
    }

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        if self.input == from {
            self.input = to;
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
pub enum AddOrSub {
    Add(Add),
    Sub(Sub),
}

impl AddOrSub {
    pub const fn lhs(&self) -> InputPort {
        match self {
            Self::Add(add) => add.lhs(),
            Self::Sub(sub) => sub.lhs(),
        }
    }

    pub const fn rhs(&self) -> InputPort {
        match self {
            Self::Add(add) => add.rhs(),
            Self::Sub(sub) => sub.rhs(),
        }
    }

    pub const fn value(&self) -> OutputPort {
        match self {
            Self::Add(add) => add.value(),
            Self::Sub(sub) => sub.value(),
        }
    }

    pub fn cast_output_dest(graph: &Rvsdg, output: OutputPort) -> Option<Self> {
        graph.output_dest_node(output)?.try_into().ok()
    }

    pub fn cast_input_source(graph: &Rvsdg, input: InputPort) -> Option<Self> {
        graph.input_source_node(input).try_into().ok()
    }

    /// Returns `true` if the add or sub is an [`Add`].
    ///
    /// [`Add`]: AddOrSub::Add
    pub const fn is_add(&self) -> bool {
        matches!(self, Self::Add(..))
    }

    /// Returns `true` if the add or sub is a [`Sub`].
    ///
    /// [`Sub`]: AddOrSub::Sub
    #[allow(dead_code)]
    pub const fn is_sub(&self) -> bool {
        matches!(self, Self::Sub(..))
    }
}

impl TryFrom<Node> for AddOrSub {
    type Error = Node;

    fn try_from(node: Node) -> Result<Self, Self::Error> {
        match node {
            Node::Add(add) => Ok(Self::Add(add)),
            Node::Sub(sub) => Ok(Self::Sub(sub)),
            node => Err(node),
        }
    }
}

impl<'a> TryFrom<&'a Node> for AddOrSub {
    type Error = &'a Node;

    fn try_from(node: &'a Node) -> Result<Self, Self::Error> {
        match node {
            &Node::Add(add) => Ok(Self::Add(add)),
            &Node::Sub(sub) => Ok(Self::Sub(sub)),
            node => Err(node),
        }
    }
}
