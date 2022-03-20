use crate::values::Ptr;
use cranelift::{codegen::ir::immediates::Offset32, prelude::Imm64};
use std::{
    fmt::{self, Debug, Display, LowerHex, UpperHex},
    num::Wrapping,
    ops::{Add, Mul, Not, Sub},
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Cell {
    value: u8,
}

impl Cell {
    pub const MAX: Self = Self::new(u8::MAX);
    pub const MIN: Self = Self::new(u8::MIN);

    #[inline]
    pub const fn new(value: u8) -> Self {
        Self { value }
    }

    #[inline]
    pub const fn zero() -> Self {
        Self::new(0)
    }

    #[inline]
    pub const fn one() -> Self {
        Self::new(1)
    }

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.value == 0
    }

    #[inline]
    pub const fn into_inner(self) -> u8 {
        self.value
    }

    #[inline]
    pub const fn wrapping_add(self, other: Self) -> Self {
        Self::new(self.value.wrapping_add(other.value))
    }

    #[inline]
    pub const fn wrapping_sub(self, other: Self) -> Self {
        Self::new(self.value.wrapping_sub(other.value))
    }

    #[inline]
    pub const fn wrapping_mul(self, other: Self) -> Self {
        Self::new(self.value.wrapping_mul(other.value))
    }

    #[inline]
    pub const fn wrapping_neg(self) -> Self {
        Self::new(self.value.wrapping_neg())
    }

    #[inline]
    pub const fn into_ptr(self, tape_len: u16) -> Ptr {
        Ptr::new(self.value as u16, tape_len)
    }
}

impl PartialEq<u8> for Cell {
    #[inline]
    fn eq(&self, other: &u8) -> bool {
        self.value == *other
    }
}

impl PartialEq<Cell> for u8 {
    #[inline]
    fn eq(&self, other: &Cell) -> bool {
        *self == other.value
    }
}

impl From<Ptr> for Cell {
    #[inline]
    fn from(ptr: Ptr) -> Self {
        ptr.into_cell()
    }
}

impl From<u8> for Cell {
    #[inline]
    fn from(byte: u8) -> Self {
        Self::new(byte)
    }
}

impl From<u16> for Cell {
    #[inline]
    fn from(int: u16) -> Self {
        Self::new(int.rem_euclid(256) as u8)
    }
}

impl From<i32> for Cell {
    #[inline]
    #[track_caller]
    fn from(int: i32) -> Self {
        Self::new(int.try_into().expect("failed to convert i32 into u8"))
    }
}

impl From<Wrapping<u16>> for Cell {
    #[inline]
    fn from(int: Wrapping<u16>) -> Self {
        Self::from(int.0)
    }
}

impl From<Cell> for u8 {
    #[inline]
    fn from(cell: Cell) -> Self {
        cell.value
    }
}

impl From<Cell> for Imm64 {
    #[inline]
    fn from(cell: Cell) -> Self {
        Imm64::new(cell.value as i64)
    }
}

impl From<Cell> for Offset32 {
    #[inline]
    fn from(cell: Cell) -> Self {
        Offset32::new(cell.value as i32)
    }
}

impl Add for Cell {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.wrapping_add(rhs)
    }
}

impl Add<&Self> for Cell {
    type Output = Self;

    #[inline]
    fn add(self, &rhs: &Self) -> Self::Output {
        self.wrapping_add(rhs)
    }
}

impl Add<Cell> for &Cell {
    type Output = Cell;

    #[inline]
    fn add(self, rhs: Cell) -> Self::Output {
        self.wrapping_add(rhs)
    }
}

impl Add<Self> for &Cell {
    type Output = Cell;

    #[inline]
    fn add(self, &rhs: Self) -> Self::Output {
        self.wrapping_add(rhs)
    }
}

impl Add<Ptr> for Cell {
    type Output = Ptr;

    #[inline]
    fn add(self, rhs: Ptr) -> Self::Output {
        self.into_ptr(rhs.tape_len()).wrapping_add(rhs)
    }
}

impl Add<&Ptr> for Cell {
    type Output = Ptr;

    #[inline]
    fn add(self, &rhs: &Ptr) -> Self::Output {
        self.into_ptr(rhs.tape_len()).wrapping_add(rhs)
    }
}

impl Add<Ptr> for &Cell {
    type Output = Ptr;

    #[inline]
    fn add(self, rhs: Ptr) -> Self::Output {
        self.into_ptr(rhs.tape_len()).wrapping_add(rhs)
    }
}

impl Add<&Ptr> for &Cell {
    type Output = Ptr;

    #[inline]
    fn add(self, &rhs: &Ptr) -> Self::Output {
        self.into_ptr(rhs.tape_len()).wrapping_add(rhs)
    }
}

impl Sub for Cell {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.wrapping_sub(rhs)
    }
}

impl Sub<&Self> for Cell {
    type Output = Self;

    #[inline]
    fn sub(self, &rhs: &Self) -> Self::Output {
        self.wrapping_sub(rhs)
    }
}

impl Sub<Cell> for &Cell {
    type Output = Cell;

    #[inline]
    fn sub(self, rhs: Cell) -> Self::Output {
        self.wrapping_sub(rhs)
    }
}

impl Sub<Self> for &Cell {
    type Output = Cell;

    #[inline]
    fn sub(self, &rhs: Self) -> Self::Output {
        self.wrapping_sub(rhs)
    }
}

impl Sub<Ptr> for Cell {
    type Output = Ptr;

    #[inline]
    fn sub(self, rhs: Ptr) -> Self::Output {
        self.into_ptr(rhs.tape_len()).wrapping_sub(rhs)
    }
}

impl Sub<&Ptr> for Cell {
    type Output = Ptr;

    #[inline]
    fn sub(self, &rhs: &Ptr) -> Self::Output {
        self.into_ptr(rhs.tape_len()).wrapping_sub(rhs)
    }
}

impl Sub<Ptr> for &Cell {
    type Output = Ptr;

    #[inline]
    fn sub(self, rhs: Ptr) -> Self::Output {
        self.into_ptr(rhs.tape_len()).wrapping_sub(rhs)
    }
}

impl Mul for Cell {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.wrapping_mul(rhs)
    }
}

impl Mul<&Cell> for Cell {
    type Output = Self;

    #[inline]
    fn mul(self, &rhs: &Self) -> Self::Output {
        self.wrapping_mul(rhs)
    }
}

impl Mul<Ptr> for Cell {
    type Output = Ptr;

    #[inline]
    fn mul(self, rhs: Ptr) -> Self::Output {
        self.into_ptr(rhs.tape_len()).wrapping_mul(rhs)
    }
}

impl Mul<&Ptr> for Cell {
    type Output = Ptr;

    #[inline]
    fn mul(self, &rhs: &Ptr) -> Self::Output {
        self.into_ptr(rhs.tape_len()).wrapping_mul(rhs)
    }
}

impl Not for Cell {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        Self::new(!self.value)
    }
}

impl Not for &Cell {
    type Output = Cell;

    #[inline]
    fn not(self) -> Self::Output {
        Cell::new(!self.value)
    }
}

impl Debug for Cell {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.value, f)
    }
}

impl Display for Cell {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.value, f)
    }
}

impl LowerHex for Cell {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        LowerHex::fmt(&self.value, f)
    }
}

impl UpperHex for Cell {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        UpperHex::fmt(&self.value, f)
    }
}

#[cfg(test)]
mod tests {
    use crate::values::Cell;

    #[test]
    fn from_u16() {
        let zero = 0u16;
        assert_eq!(Cell::from(zero), Cell::zero());

        let two_fifty_five = 255u16;
        assert_eq!(Cell::from(two_fifty_five), Cell::new(255));

        let two_fifty_six = 256u16;
        assert_eq!(Cell::from(two_fifty_six), Cell::new(0));
    }
}
