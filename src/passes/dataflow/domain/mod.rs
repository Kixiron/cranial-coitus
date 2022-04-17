mod bitmap_u16;
mod byte_set;
mod int_set;
mod utils;

pub use byte_set::{ByteSet, ProgramTape};
pub use int_set::IntSet;

use crate::{
    ir::Const,
    values::{Cell, Ptr},
};
use std::{
    fmt::{self, Debug, Display},
    hint::unreachable_unchecked,
    mem::swap,
};

#[derive(Debug, Clone)]
pub enum Domain {
    Bool(BoolSet),
    Byte(ByteSet),
    Int(IntSet),
}

impl Domain {
    pub fn as_bool_set(&self) -> Option<BoolSet> {
        if let Self::Bool(booleans) = *self {
            Some(booleans)
        } else {
            None
        }
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn into_byte_set(&self) -> ByteSet {
        match self {
            Self::Byte(bytes) => *bytes,
            Self::Int(ints) => {
                let mut bytes = ByteSet::empty();
                for int in ints.iter() {
                    bytes.insert(int.into_cell().into_inner());
                }

                bytes
            }
            Self::Bool(_) => unreachable!(),
        }
    }

    pub fn union(&mut self, other: &Domain) -> bool {
        match (self, other) {
            (Self::Bool(lhs), Self::Bool(rhs)) => lhs.union(rhs),
            (Self::Byte(lhs), Self::Byte(rhs)) => lhs.union(*rhs),
            (Self::Int(lhs), Self::Int(rhs)) => lhs.union(rhs),

            // TODO: Efficiency
            (Self::Int(lhs), Self::Byte(rhs)) => lhs.union_bytes(*rhs),
            (lhs @ Self::Byte(_), Self::Int(rhs)) => {
                let lhs_bytes = if let Self::Byte(bytes) = lhs {
                    *bytes
                } else if cfg!(debug_assertions) {
                    unreachable!()
                } else {
                    unsafe { unreachable_unchecked() }
                };

                let mut rhs = rhs.clone();
                let changed = rhs.union_bytes(lhs_bytes);
                *lhs = Self::Int(rhs);

                changed
            }

            (Self::Bool(_), _) | (_, Self::Bool(_)) => unreachable!(),
        }
    }

    pub fn union_mut(&mut self, other: &mut Domain) -> bool {
        match (self, other) {
            (Self::Bool(lhs), Self::Bool(rhs)) => lhs.union(rhs),
            (Self::Byte(lhs), Self::Byte(rhs)) => lhs.union(*rhs),
            (Self::Int(lhs), Self::Int(rhs)) => lhs.union(rhs),

            (Self::Int(lhs), Self::Byte(rhs)) => lhs.union_bytes(*rhs),
            (lhs @ Self::Byte(_), rhs @ Self::Int(_)) => {
                let lhs_bytes = if let Self::Byte(bytes) = lhs {
                    *bytes
                } else if cfg!(debug_assertions) {
                    unreachable!()
                } else {
                    unsafe { unreachable_unchecked() }
                };

                let changed = if let Self::Int(rhs) = rhs {
                    rhs.union_bytes(lhs_bytes)
                } else if cfg!(debug_assertions) {
                    unreachable!()
                } else {
                    // Safety: The match guard ensures this is always an Int
                    unsafe { unreachable_unchecked() }
                };

                swap(lhs, rhs);
                changed
            }

            (Self::Bool(_), _) | (_, Self::Bool(_)) => unreachable!(),
        }
    }

    pub fn intersects(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bool(lhs), Self::Bool(rhs)) => lhs.intersects(*rhs),
            (Self::Byte(lhs), Self::Byte(rhs)) => lhs.intersects(*rhs),
            (Self::Int(lhs), Self::Int(rhs)) => lhs.intersects(rhs),

            (Self::Int(ints), Self::Byte(bytes)) | (Self::Byte(bytes), Self::Int(ints)) => {
                ints.intersects_bytes(*bytes)
            }

            (Self::Bool(_), _) | (_, Self::Bool(_)) => unreachable!(),
        }
    }

    pub fn intersect(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bool(lhs), Self::Bool(rhs)) => Self::Bool(lhs.intersect(*rhs)),
            (Self::Byte(lhs), Self::Byte(rhs)) => Self::Byte(lhs.intersection(rhs)),
            (Self::Int(lhs), Self::Int(rhs)) => Self::Int(lhs.intersect(rhs)),

            (Self::Int(ints), Self::Byte(bytes)) | (Self::Byte(bytes), Self::Int(ints)) => {
                Self::Int(ints.intersect_bytes(bytes))
            }

            (Self::Bool(_), _) | (_, Self::Bool(_)) => unreachable!(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Bool(booleans) => booleans.is_empty(),
            Self::Byte(bytes) => bytes.is_empty(),
            Self::Int(ints) => ints.is_empty(),
        }
    }

    pub fn as_singleton(&self) -> Option<Const> {
        match self {
            Self::Bool(booleans) => booleans.as_singleton().map(Const::Bool),
            Self::Byte(bytes) => bytes
                .as_singleton()
                .map(|byte| Const::Cell(Cell::new(byte))),
            Self::Int(ints) => ints.as_singleton().map(Const::Ptr),
        }
    }
}

impl Default for Domain {
    fn default() -> Self {
        Self::Bool(BoolSet::empty())
    }
}

impl From<BoolSet> for Domain {
    fn from(booleans: BoolSet) -> Self {
        Self::Bool(booleans)
    }
}

impl From<ByteSet> for Domain {
    fn from(bytes: ByteSet) -> Self {
        Self::Byte(bytes)
    }
}

impl From<IntSet> for Domain {
    fn from(ints: IntSet) -> Self {
        Self::Int(ints)
    }
}

impl From<bool> for Domain {
    fn from(bool: bool) -> Self {
        Self::Bool(BoolSet::from(bool))
    }
}

impl From<u8> for Domain {
    fn from(byte: u8) -> Self {
        Self::Byte(ByteSet::from(byte))
    }
}

impl From<Cell> for Domain {
    fn from(cell: Cell) -> Self {
        Self::Byte(ByteSet::from(cell))
    }
}

impl From<Ptr> for Domain {
    fn from(ptr: Ptr) -> Self {
        Self::Int(IntSet::from(ptr))
    }
}

impl Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(bool) => Debug::fmt(bool, f),
            Self::Byte(byte) => Debug::fmt(byte, f),
            Self::Int(int) => Debug::fmt(int, f),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct BoolSet {
    /// A bitset of values the boolean can inhabit
    values: u8,
}

impl BoolSet {
    /// Creates a boolean set with zero inhabitants (neither `true` nor `false`)
    pub const fn empty() -> Self {
        Self { values: 0 }
    }

    /// Creates a boolean set with all inhabitants (both `true` and `false`)
    pub const fn full() -> Self {
        Self::empty().with_value(true).with_value(false)
    }

    pub const fn singleton(value: bool) -> Self {
        Self::empty().with_value(value)
    }

    pub const fn is_empty(&self) -> bool {
        self.values == Self::empty().values
    }

    pub const fn contains(&self, value: bool) -> bool {
        (self.values & (1 << value as usize)) != 0
    }

    pub fn add(&mut self, value: bool) {
        self.values |= 1 << value as usize;
    }

    pub const fn with_value(&self, value: bool) -> Self {
        Self {
            values: self.values | (1 << value as usize),
        }
    }

    pub fn remove(&mut self, value: bool) {
        self.values &= !(1 << value as usize);
    }

    pub fn union(&mut self, other: &Self) -> bool {
        let old = self.values;
        self.values |= other.values;
        self.values != old
    }

    pub fn intersect(&self, other: Self) -> Self {
        Self {
            values: self.values & other.values,
        }
    }

    pub fn intersects(&self, other: Self) -> bool {
        self.intersect(other) != Self::empty()
    }

    pub fn as_singleton(&self) -> Option<bool> {
        if *self == Self::singleton(true) {
            Some(true)
        } else if *self == Self::singleton(false) {
            Some(false)
        } else {
            None
        }
    }

    pub fn as_slice(&self) -> &[bool] {
        match (self.contains(true), self.contains(false)) {
            (false, false) => &[],
            (true, false) => &[true],
            (false, true) => &[false],
            (true, true) => &[true, false],
        }
    }
}

impl From<bool> for BoolSet {
    fn from(value: bool) -> Self {
        Self::empty().with_value(value)
    }
}

impl Debug for BoolSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.as_slice()).finish()
    }
}

impl Display for BoolSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

// Calculates the filtered cartesian product of both domains where the values are inequal
// (a, b) := { (a, b) ∈ A × B | A ≠ B }
pub fn differential_product(lhs: &Domain, rhs: &Domain) -> (Domain, Domain) {
    match (lhs, rhs) {
        (Domain::Bool(lhs), Domain::Bool(rhs)) => {
            let (mut lhs_false, mut rhs_false) = (BoolSet::empty(), BoolSet::empty());
            for &lhs in lhs.as_slice() {
                for &rhs in rhs.as_slice() {
                    if lhs != rhs {
                        lhs_false.add(lhs);
                        rhs_false.add(rhs);
                    }
                }
            }

            (Domain::Bool(lhs_false), Domain::Bool(rhs_false))
        }

        (Domain::Byte(lhs), Domain::Byte(rhs)) => {
            let (mut lhs_false, mut rhs_false) = (ByteSet::empty(), ByteSet::empty());
            for lhs in lhs.iter() {
                for rhs in rhs.iter() {
                    if lhs != rhs {
                        lhs_false.insert(lhs);
                        rhs_false.insert(rhs);
                    }
                }
            }

            (Domain::Byte(lhs_false), Domain::Byte(rhs_false))
        }

        (Domain::Int(lhs), Domain::Int(rhs)) => {
            let (mut lhs_false, mut rhs_false) =
                (IntSet::empty(lhs.tape_len), IntSet::empty(lhs.tape_len));
            for lhs in lhs.iter() {
                for rhs in rhs.iter() {
                    if lhs != rhs {
                        lhs_false.add(lhs);
                        rhs_false.add(rhs);
                    }
                }
            }

            (Domain::Int(lhs_false), Domain::Int(rhs_false))
        }

        (Domain::Int(lhs), Domain::Byte(rhs)) | (Domain::Byte(rhs), Domain::Int(lhs)) => {
            let (mut lhs_false, mut rhs_false) =
                (IntSet::empty(lhs.tape_len), IntSet::empty(lhs.tape_len));
            for lhs in lhs.iter() {
                for rhs in rhs.iter() {
                    let rhs = Ptr::new(rhs as u16, lhs.tape_len());

                    if lhs != rhs {
                        lhs_false.add(lhs);
                        rhs_false.add(rhs);
                    }
                }
            }

            (Domain::Int(lhs_false), Domain::Int(rhs_false))
        }

        (Domain::Bool(_), _) | (_, Domain::Bool(_)) => unreachable!(),
    }
}
