use std::fmt::{self, Debug, Display, Write};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct NodeId(pub u32);

impl NodeId {
    pub(in crate::graph) const fn new(id: u32) -> Self {
        Self(id)
    }
}

impl Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NodeId(")?;
        Debug::fmt(&self.0, f)?;
        f.write_char(')')
    }
}

impl Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_char('n')?;
        Display::fmt(&self.0, f)
    }
}
