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
    pub(super) const fn new(id: u32) -> Self {
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
    pub(super) const fn new(id: PortId) -> Self {
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
