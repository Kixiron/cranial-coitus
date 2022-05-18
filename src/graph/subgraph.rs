use crate::graph::{End, NodeId, Rvsdg, Start};

#[derive(Debug, Clone, PartialEq)]
pub struct Subgraph {
    pub(super) start: NodeId,
    pub(super) end: NodeId,
}

impl Subgraph {
    pub(super) fn new(start: NodeId, end: NodeId) -> Self {
        Self { start, end }
    }

    /// Get the [`Start`] of the subgraph
    pub fn start_node(&self, graph: &Rvsdg) -> Start {
        *graph.to_node(self.start)
    }

    /// Get the [`End`] of the subgraph
    pub fn end_node(&self, graph: &Rvsdg) -> End {
        *graph.to_node(self.end)
    }
}
