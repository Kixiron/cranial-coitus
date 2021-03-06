use crate::{
    graph::{Node, Rvsdg},
    utils::PerfEvent,
};
use std::ops::Neg;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub branches: usize,
    pub loops: usize,
    pub loads: usize,
    pub stores: usize,
    pub constants: usize,
    pub instructions: usize,
    pub io_ops: usize,
    pub scans: usize,
}

impl Stats {
    pub fn new() -> Self {
        Self {
            branches: 0,
            loops: 0,
            loads: 0,
            stores: 0,
            constants: 0,
            instructions: 0,
            io_ops: 0,
            scans: 0,
        }
    }

    pub fn difference(&self, new: Self) -> StatsChange {
        let diff = |old, new| {
            let diff = (((old as f64 - new as f64) / old as f64) * 100.0).neg();

            if diff.is_nan() || diff == -0.0 {
                0.0
            } else if diff.is_infinite() {
                if diff.is_sign_positive() {
                    100.0
                } else {
                    -100.0
                }
            } else {
                diff
            }
        };

        StatsChange {
            branches: diff(self.branches, new.branches),
            loops: diff(self.loops, new.loops),
            loads: diff(self.loads, new.loads),
            stores: diff(self.stores, new.stores),
            constants: diff(self.constants, new.constants),
            instructions: diff(self.instructions, new.instructions),
            io_ops: diff(self.io_ops, new.io_ops),
            scans: diff(self.scans, new.scans),
        }
    }
}

impl Default for Stats {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StatsChange {
    pub branches: f64,
    pub loops: f64,
    pub loads: f64,
    pub stores: f64,
    pub constants: f64,
    pub instructions: f64,
    pub io_ops: f64,
    pub scans: f64,
}

impl Rvsdg {
    pub fn stats(&self) -> Stats {
        let _event = PerfEvent::new("collect-graph-stats");

        let mut stats = Stats::new();
        self.for_each_transitive_node(|_node_id, node| match node {
            Node::Int(_, _) | Node::Byte(_, _) | Node::Bool(_, _) => stats.constants += 1,
            Node::Add(_)
            | Node::Sub(_)
            | Node::Mul(_)
            | Node::Eq(_)
            | Node::Neq(_)
            | Node::Not(_)
            | Node::Neg(_) => {
                stats.instructions += 1;
            }
            Node::Load(_) => {
                stats.instructions += 1;
                stats.loads += 1;
            }
            Node::Store(_) => {
                stats.instructions += 1;
                stats.stores += 1;
            }
            Node::Scan(_) => {
                stats.instructions += 1;
                stats.scans += 1;
            }
            Node::Input(_) | Node::Output(_) => {
                stats.instructions += 1;
                stats.io_ops += 1;
            }
            Node::Theta(_) => {
                stats.instructions += 1;
                stats.loops += 1;
            }
            Node::Gamma(_) => {
                stats.instructions += 1;
                stats.branches += 1;
            }

            Node::Start(_) | Node::End(_) | Node::InputParam(_) | Node::OutputParam(_) => {}
        });

        stats
    }
}
