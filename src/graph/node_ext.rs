use crate::graph::{InputPort, NodeId, OutputPort};
use tinyvec::TinyVec;

pub trait NodeExt {
    fn node(&self) -> NodeId;

    fn input_desc(&self) -> EdgeDescriptor;

    fn all_input_ports(&self) -> TinyVec<[InputPort; 4]>;

    fn output_desc(&self) -> EdgeDescriptor;

    fn all_output_ports(&self) -> TinyVec<[OutputPort; 4]>;
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeDescriptor {
    effect: EdgeCount,
    value: EdgeCount,
}

impl EdgeDescriptor {
    pub const fn new(effect: EdgeCount, value: EdgeCount) -> Self {
        Self { effect, value }
    }

    pub const fn effect(&self) -> EdgeCount {
        self.effect
    }

    pub const fn value(&self) -> EdgeCount {
        self.value
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeCount {
    min: Option<usize>,
    max: Option<usize>,
}

impl EdgeCount {
    pub const fn new(min: Option<usize>, max: Option<usize>) -> Self {
        Self { min, max }
    }

    pub const fn unlimited() -> Self {
        Self::new(None, None)
    }

    pub const fn exact(count: usize) -> Self {
        Self::new(Some(count), Some(count))
    }

    pub const fn zero() -> Self {
        Self::exact(0)
    }

    pub const fn one() -> Self {
        Self::exact(1)
    }

    pub const fn two() -> Self {
        Self::exact(2)
    }

    #[allow(dead_code)]
    pub const fn three() -> Self {
        Self::exact(3)
    }

    pub const fn contains(&self, value: usize) -> bool {
        match (self.min, self.max) {
            (Some(min), None) => value >= min,
            (None, Some(max)) => value <= max,
            (Some(min), Some(max)) => min <= value && value <= max,
            (None, None) => true,
        }
    }

    pub const fn min(&self) -> Option<usize> {
        self.min
    }

    pub const fn max(&self) -> Option<usize> {
        self.max
    }
}
