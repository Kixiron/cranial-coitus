use std::fmt::{self, Display};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Effect,
    Value,
}

impl EdgeKind {
    /// Returns `true` if the edge kind is [`Effect`].
    ///
    /// [`Effect`]: EdgeKind::Effect
    pub const fn is_effect(&self) -> bool {
        matches!(self, Self::Effect)
    }

    /// Returns `true` if the edge kind is [`Value`].
    ///
    /// [`Value`]: EdgeKind::Value
    pub const fn is_value(&self) -> bool {
        matches!(self, Self::Value)
    }
}

impl Default for EdgeKind {
    fn default() -> Self {
        Self::Value
    }
}

impl Display for EdgeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Effect => f.write_str("effect"),
            Self::Value => f.write_str("value"),
        }
    }
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

    pub const fn zero() -> Self {
        Self::new(EdgeCount::zero(), EdgeCount::zero())
    }

    pub const fn from_values(values: usize) -> Self {
        Self::new(EdgeCount::zero(), EdgeCount::exact(values))
    }

    pub const fn from_effects(effects: usize) -> Self {
        Self::new(EdgeCount::exact(effects), EdgeCount::zero())
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

    pub const fn at_least(minimum: usize) -> Self {
        Self::new(Some(minimum), None)
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
