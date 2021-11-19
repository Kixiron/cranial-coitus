use crate::graph::{End, NodeId, Rvsdg, Start};
use std::ops::{Deref, DerefMut};

#[derive(Debug, Clone, PartialEq)]
pub struct Subgraph {
    pub(super) graph: Rvsdg,
    pub(super) start: NodeId,
    pub(super) end: NodeId,
}

impl Subgraph {
    pub(super) fn new(graph: Rvsdg, start: NodeId, end: NodeId) -> Self {
        Self { graph, start, end }
    }

    /// Get the [`Start`] of the subgraph
    pub fn start_node(&self) -> Start {
        *self.graph.to_node(self.start)
    }

    /// Get the [`End`] of the subgraph
    pub fn end_node(&self) -> End {
        *self.graph.to_node(self.end)
    }
}

impl Deref for Subgraph {
    type Target = Rvsdg;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for Subgraph {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}
