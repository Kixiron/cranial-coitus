use crate::graph::{EdgeKind, NodeId};
use std::{
    cmp::Eq,
    fmt::{self, Debug, Display, Write},
    hash::Hash,
};

pub trait Port: Debug + Clone + Copy + PartialEq + Eq + Hash {
    fn port(&self) -> PortId;

    fn raw(&self) -> u32 {
        self.port().0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct PortId(u32);

impl PortId {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    const fn inner(&self) -> u32 {
        self.0
    }
}

impl Port for PortId {
    fn port(&self) -> PortId {
        *self
    }
}

impl Debug for PortId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PortId(")?;
        Debug::fmt(&self.0, f)?;
        f.write_char(')')
    }
}

impl Display for PortId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct InputPort(PortId);

impl InputPort {
    pub(super) const fn new(id: PortId) -> Self {
        Self(id)
    }
}

impl Port for InputPort {
    fn port(&self) -> PortId {
        self.0
    }
}

impl Debug for InputPort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("InputPort(")?;
        Debug::fmt(&self.0.inner(), f)?;
        f.write_char(')')
    }
}

impl Display for InputPort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0.inner(), f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct OutputPort(PortId);

impl OutputPort {
    pub const fn new(id: PortId) -> Self {
        Self(id)
    }
}

impl Port for OutputPort {
    fn port(&self) -> PortId {
        self.0
    }
}

impl Debug for OutputPort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("OutputPort(")?;
        Debug::fmt(&self.0.inner(), f)?;
        f.write_char(')')
    }
}

impl Display for OutputPort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0.inner(), f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortData {
    pub kind: PortKind,
    pub edge: EdgeKind,
    pub parent: NodeId,
}

impl PortData {
    pub const fn new(kind: PortKind, edge: EdgeKind, node: NodeId) -> Self {
        Self {
            kind,
            edge,
            parent: node,
        }
    }

    pub const fn input(node: NodeId, edge: EdgeKind) -> Self {
        Self::new(PortKind::Input, edge, node)
    }

    pub const fn output(node: NodeId, edge: EdgeKind) -> Self {
        Self::new(PortKind::Output, edge, node)
    }
}

impl Display for PortData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} port for {}", self.kind, self.parent)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortKind {
    Input,
    Output,
}

impl PortKind {
    /// Returns `true` if the port kind is [`Input`].
    ///
    /// [`Input`]: PortKind::Input
    pub const fn is_input(&self) -> bool {
        matches!(self, Self::Input)
    }

    /// Returns `true` if the port kind is [`Output`].
    ///
    /// [`Output`]: PortKind::Output
    pub const fn is_output(&self) -> bool {
        matches!(self, Self::Output)
    }
}

impl Display for PortKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input => f.write_str("input"),
            Self::Output => f.write_str("output"),
        }
    }
}
