use crate::{
    ir::Const,
    values::{Cell, Ptr},
};
use roaring::RoaringBitmap;
use std::{
    borrow::Cow,
    fmt::{self, Debug, Display},
    mem::take,
};

#[derive(Debug, Clone)]
pub enum NormalizedDomains {
    Bool(BoolSet, BoolSet),
    Byte(ByteSet, ByteSet),
    Int(IntSet, IntSet),
}

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
                    bytes.add(int.into_cell().into_inner());
                }

                bytes
            }
            Self::Bool(_) => unreachable!(),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn into_int_set(&self, tape_len: u16) -> Cow<'_, IntSet> {
        match self {
            Self::Byte(bytes) => Cow::Owned(bytes.into_int_set(tape_len)),
            Self::Int(ints) => Cow::Borrowed(ints),
            Self::Bool(_) => unreachable!(),
        }
    }

    pub fn union(&mut self, other: &Domain) {
        match (self, other) {
            (Self::Bool(lhs), Self::Bool(rhs)) => lhs.union(rhs),
            (Self::Byte(lhs), Self::Byte(rhs)) => lhs.union(*rhs),
            (Self::Int(lhs), Self::Int(rhs)) => lhs.union(rhs),

            // TODO: Efficiency
            (Self::Int(lhs), Self::Byte(rhs)) => {
                let mut rhs = rhs.into_int_set(lhs.tape_len);
                lhs.union_mut(&mut rhs);
            }
            (lhs @ Self::Byte(_), Self::Int(rhs)) => {
                let mut lhs_set = if let Cow::Owned(lhs) = lhs.into_int_set(rhs.tape_len) {
                    lhs
                } else {
                    unreachable!()
                };
                lhs_set.union(rhs);

                *lhs = Self::Int(lhs_set);
            }

            (Self::Bool(_), _) | (_, Self::Bool(_)) => unreachable!(),
        }
    }

    pub fn union_mut(&mut self, other: &mut Domain) {
        match (self, other) {
            (Self::Bool(lhs), Self::Bool(rhs)) => lhs.union(rhs),
            (Self::Byte(lhs), Self::Byte(rhs)) => lhs.union_mut(rhs),
            (Self::Int(lhs), Self::Int(rhs)) => lhs.union_mut(rhs),

            (Self::Int(lhs), Self::Byte(rhs)) => {
                let mut rhs = rhs.into_int_set(lhs.tape_len);
                lhs.union_mut(&mut rhs);
            }
            (lhs @ Self::Byte(_), Self::Int(rhs)) => {
                let mut lhs_set = if let Cow::Owned(lhs) = lhs.into_int_set(rhs.tape_len) {
                    lhs
                } else {
                    unreachable!()
                };
                lhs_set.union_mut(rhs);

                *lhs = Self::Int(lhs_set);
            }

            (Self::Bool(_), _) | (_, Self::Bool(_)) => unreachable!(),
        }
    }

    pub fn intersect(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bool(lhs), Self::Bool(rhs)) => Self::Bool(lhs.intersect(*rhs)),
            (Self::Byte(lhs), Self::Byte(rhs)) => Self::Byte(lhs.intersect(rhs)),
            (Self::Int(lhs), Self::Int(rhs)) => Self::Int(lhs.intersect(rhs)),

            (Self::Int(ints), Self::Byte(bytes)) | (Self::Byte(bytes), Self::Int(ints)) => {
                Self::Int(bytes.into_int_set(ints.tape_len).intersect_ref(ints))
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

    pub(crate) fn normalize(self, other: Self) -> NormalizedDomains {
        match (self, other) {
            (Self::Bool(lhs), Self::Bool(rhs)) => NormalizedDomains::Bool(lhs, rhs),
            (Self::Byte(lhs), Self::Byte(rhs)) => NormalizedDomains::Byte(lhs, rhs),
            (Self::Int(lhs), Self::Int(rhs)) => NormalizedDomains::Int(lhs, rhs),

            (Self::Int(lhs), Self::Byte(rhs)) => {
                let tape_len = lhs.tape_len;
                NormalizedDomains::Int(lhs, rhs.into_int_set(tape_len))
            }
            (Self::Byte(lhs), Self::Int(rhs)) => {
                NormalizedDomains::Int(lhs.into_int_set(rhs.tape_len), rhs)
            }

            (Self::Bool(_), _) | (_, Self::Bool(_)) => unreachable!(),
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

    pub const fn is_empty(&self) -> bool {
        self.values == Self::empty().values
    }

    /// Creates a boolean set with all inhabitants (both `true` and `false`)
    pub const fn full() -> Self {
        Self::empty().with_value(true).with_value(false)
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

    pub fn union(&mut self, other: &Self) {
        self.values |= other.values;
    }

    pub fn intersect(&self, other: Self) -> Self {
        Self {
            values: self.values & other.values,
        }
    }

    pub fn as_singleton(&self) -> Option<bool> {
        match (self.contains(true), self.contains(false)) {
            (true, true) | (false, false) => None,
            (true, false) => Some(true),
            (false, true) => Some(false),
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

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct ByteSet {
    values: byte_set::ByteSet,
}

impl ByteSet {
    pub fn empty() -> Self {
        Self {
            values: byte_set::ByteSet::new(),
        }
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }

    pub fn singleton(value: u8) -> Self {
        Self {
            values: byte_set::ByteSet::from_byte(value),
        }
    }

    pub fn full() -> Self {
        Self {
            values: byte_set::ByteSet::full(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn add(&mut self, value: u8) {
        self.values.insert(value);
    }

    pub fn make_full(&mut self) {
        *self = Self::full();
    }

    pub fn union(&mut self, other: Self) {
        // FIXME: Remove this clone
        self.values = self.values.union(other.values);
    }

    pub(crate) fn union_mut(&mut self, other: &mut ByteSet) {
        self.values = take(&mut self.values).union(take(&mut other.values));
    }

    pub fn as_singleton(&self) -> Option<u8> {
        (self.values.len() == 1).then(|| self.values.first().unwrap())
    }

    pub fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        self.values.into_iter()
    }

    pub fn intersect(&self, other: &Self) -> Self {
        Self {
            // FIXME: Efficiency
            values: self.values.intersection(other.values),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn into_int_set(&self, tape_len: u16) -> IntSet {
        let mut ints = IntSet::empty(tape_len);
        // TODO: Efficiency
        for byte in self.iter() {
            ints.add(Ptr::new(byte as u16, tape_len));
        }

        ints
    }

    pub fn intersects(&self, other: &Self) -> bool {
        self.values.contains_any(&other.values)
    }
}

impl From<u8> for ByteSet {
    fn from(byte: u8) -> Self {
        Self::singleton(byte)
    }
}

impl From<Cell> for ByteSet {
    fn from(cell: Cell) -> Self {
        Self::singleton(cell.into_inner())
    }
}

impl Debug for ByteSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl Display for ByteSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

#[derive(Clone, PartialEq, Default)]
pub struct IntSet {
    values: RoaringBitmap,
    tape_len: u16,
}

impl IntSet {
    pub fn empty(tape_len: u16) -> Self {
        Self {
            values: RoaringBitmap::new(),
            tape_len,
        }
    }

    pub fn full(tape_len: u16) -> Self {
        let mut this = Self::empty(tape_len);
        this.values.insert_range(u16::MIN as u32..=u16::MAX as u32);
        this
    }

    pub fn is_full(&self) -> bool {
        todo!()
    }

    pub fn singleton(value: Ptr) -> Self {
        let mut this = Self::empty(value.tape_len());
        this.values.insert(value.value() as u32);
        this
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn add(&mut self, value: Ptr) {
        debug_assert_eq!(self.tape_len, value.tape_len());
        self.values.insert(value.value() as u32);
    }

    fn union(&mut self, other: &Self) {
        self.values &= &other.values;
    }

    pub fn union_mut(&mut self, other: &mut Self) {
        self.values &= take(&mut other.values);
    }

    pub fn as_singleton(&self) -> Option<Ptr> {
        (self.values.len() == 1).then(|| self.iter().next().unwrap())
    }

    pub fn iter(&self) -> impl Iterator<Item = Ptr> + '_ {
        self.values
            .iter()
            .map(|value| Ptr::new(value as u16, self.tape_len))
    }

    pub fn intersect(&self, other: &Self) -> Self {
        debug_assert_eq!(self.tape_len, other.tape_len);

        Self {
            values: &self.values | &other.values,
            tape_len: self.tape_len,
        }
    }

    pub fn intersect_ref(self, other: &Self) -> Self {
        debug_assert_eq!(self.tape_len, other.tape_len);

        Self {
            values: self.values | &other.values,
            tape_len: self.tape_len,
        }
    }
}

impl From<Ptr> for IntSet {
    fn from(int: Ptr) -> Self {
        Self::singleton(int)
    }
}

impl Debug for IntSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl Display for IntSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
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
                        lhs_false.add(lhs);
                        rhs_false.add(rhs);
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
            let rhs = rhs.into_int_set(lhs.tape_len);

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

        (Domain::Bool(_), _) | (_, Domain::Bool(_)) => unreachable!(),
    }
}
