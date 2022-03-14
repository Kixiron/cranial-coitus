use crate::values::Cell;
use cranelift::{codegen::ir::immediates::Offset32, prelude::Imm64};
use std::{
    fmt::{self, Debug, Display, LowerHex, UpperHex},
    ops::{Add, Index, IndexMut, Mul, Not, Sub},
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Ptr {
    // Bits 1-16 hold the pointer, 17-32 hold the tape length
    // Invariant: The pointer should always be >0 and <tape_len
    value: u32,
}

impl Ptr {
    #[inline]
    pub const fn new(value: u16, tape_len: u16) -> Self {
        debug_assert!(tape_len != 0);

        Self {
            value: ((value.rem_euclid(tape_len) as u32) << 16) | tape_len as u32,
        }
    }

    #[inline]
    pub const fn value(self) -> u16 {
        (self.value >> 16) as u16
    }

    #[inline]
    pub const fn tape_len(self) -> u16 {
        self.value as u16
    }

    #[inline]
    pub const fn zero(tape_len: u16) -> Self {
        Self::new(0, tape_len)
    }

    #[inline]
    pub const fn one(tape_len: u16) -> Self {
        Self::new(1, tape_len)
    }

    /// Returns `true` if the current number is zero
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.value() == 0
    }

    /// Returns `true` if the current number is even
    #[inline]
    pub const fn is_even(self) -> bool {
        self.value() % 2 == 0
    }

    /// Returns `true` if the current number is a power of two
    #[inline]
    pub const fn is_power_of_two(self) -> bool {
        self.value().is_power_of_two()
    }

    #[inline]
    pub const fn wrapping_add(self, other: Self) -> Self {
        debug_assert!(self.tape_len() == other.tape_len());

        Self::new(
            (self.value() as i32 + other.value() as i32).rem_euclid(self.tape_len() as i32) as u16,
            self.tape_len(),
        )
    }

    #[inline]
    pub const fn wrapping_sub(self, other: Self) -> Self {
        debug_assert!(self.tape_len() == other.tape_len());

        Self::new(
            (self.value() as i32 - other.value() as i32).rem_euclid(self.tape_len() as i32) as u16,
            self.tape_len(),
        )
    }

    #[inline]
    pub const fn wrapping_mul(self, other: Self) -> Self {
        debug_assert!(self.tape_len() == other.tape_len());

        Self::new(
            (self.value() as u32 + other.value() as u32).rem_euclid(self.tape_len() as u32) as u16,
            self.tape_len(),
        )
    }

    #[inline]
    pub const fn into_cell(self) -> Cell {
        Cell::new(self.value().rem_euclid(u8::MAX as u16 + 1) as u8)
    }
}

impl From<Ptr> for Imm64 {
    #[inline]
    fn from(ptr: Ptr) -> Self {
        Imm64::new(ptr.value() as i64)
    }
}

impl From<Ptr> for Offset32 {
    #[inline]
    fn from(ptr: Ptr) -> Self {
        Offset32::new(ptr.value() as i32)
    }
}

impl<T> Index<Ptr> for [T] {
    type Output = T;

    #[inline]
    #[track_caller]
    fn index(&self, index: Ptr) -> &Self::Output {
        &self[index.value() as usize]
    }
}

impl<T> IndexMut<Ptr> for [T] {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, index: Ptr) -> &mut Self::Output {
        &mut self[index.value() as usize]
    }
}

impl<T> Index<Ptr> for Vec<T> {
    type Output = T;

    #[inline]
    #[track_caller]
    fn index(&self, index: Ptr) -> &Self::Output {
        &self[index.value() as usize]
    }
}

impl<T> IndexMut<Ptr> for Vec<T> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, index: Ptr) -> &mut Self::Output {
        &mut self[index.value() as usize]
    }
}

impl PartialEq<u16> for Ptr {
    #[inline]
    fn eq(&self, other: &u16) -> bool {
        self.value() == *other
    }
}

impl PartialEq<Ptr> for u16 {
    #[inline]
    fn eq(&self, other: &Ptr) -> bool {
        *self == other.value()
    }
}

impl Add for Ptr {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.wrapping_add(rhs)
    }
}

impl Add<&Self> for Ptr {
    type Output = Self;

    #[inline]
    fn add(self, &rhs: &Self) -> Self::Output {
        self.wrapping_add(rhs)
    }
}

impl Add<Ptr> for &Ptr {
    type Output = Ptr;

    #[inline]
    fn add(self, rhs: Ptr) -> Self::Output {
        self.wrapping_add(rhs)
    }
}

impl Add<Self> for &Ptr {
    type Output = Ptr;

    #[inline]
    fn add(self, &rhs: Self) -> Self::Output {
        self.wrapping_add(rhs)
    }
}

impl Add<Cell> for Ptr {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Cell) -> Self::Output {
        self.wrapping_add(rhs.into_ptr(self.tape_len()))
    }
}

impl Add<Cell> for &Ptr {
    type Output = Ptr;

    #[inline]
    fn add(self, rhs: Cell) -> Self::Output {
        self.wrapping_add(rhs.into_ptr(self.tape_len()))
    }
}

impl Add<&Cell> for Ptr {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &Cell) -> Self::Output {
        self.wrapping_add(rhs.into_ptr(self.tape_len()))
    }
}

impl Add<&Cell> for &Ptr {
    type Output = Ptr;

    #[inline]
    fn add(self, rhs: &Cell) -> Self::Output {
        self.wrapping_add(rhs.into_ptr(self.tape_len()))
    }
}

impl Sub for Ptr {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.wrapping_sub(rhs)
    }
}

impl Sub<&Self> for Ptr {
    type Output = Self;

    #[inline]
    fn sub(self, &rhs: &Self) -> Self::Output {
        self.wrapping_sub(rhs)
    }
}

impl Sub<Ptr> for &Ptr {
    type Output = Ptr;

    #[inline]
    fn sub(self, rhs: Ptr) -> Self::Output {
        self.wrapping_sub(rhs)
    }
}

impl Sub<Self> for &Ptr {
    type Output = Ptr;

    #[inline]
    fn sub(self, &rhs: Self) -> Self::Output {
        self.wrapping_sub(rhs)
    }
}

impl Sub<Cell> for Ptr {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Cell) -> Self::Output {
        self.wrapping_sub(rhs.into_ptr(self.tape_len()))
    }
}

impl Sub<Cell> for &Ptr {
    type Output = Ptr;

    #[inline]
    fn sub(self, rhs: Cell) -> Self::Output {
        self.wrapping_sub(rhs.into_ptr(self.tape_len()))
    }
}

impl Sub<&Cell> for Ptr {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Cell) -> Self::Output {
        self.wrapping_sub(rhs.into_ptr(self.tape_len()))
    }
}

impl Sub<&Cell> for &Ptr {
    type Output = Ptr;

    #[inline]
    fn sub(self, rhs: &Cell) -> Self::Output {
        self.wrapping_sub(rhs.into_ptr(self.tape_len()))
    }
}

impl Mul for Ptr {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.wrapping_mul(rhs)
    }
}

impl Mul<&Ptr> for Ptr {
    type Output = Self;

    #[inline]
    fn mul(self, &rhs: &Self) -> Self::Output {
        self.wrapping_mul(rhs)
    }
}

impl Mul<Cell> for Ptr {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Cell) -> Self::Output {
        self.wrapping_mul(rhs.into_ptr(self.tape_len()))
    }
}

impl Mul<&Cell> for Ptr {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Cell) -> Self::Output {
        self.wrapping_mul(rhs.into_ptr(self.tape_len()))
    }
}

impl Not for Ptr {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        Self::new(!self.value(), self.tape_len())
    }
}

impl Not for &Ptr {
    type Output = Ptr;

    #[inline]
    fn not(self) -> Self::Output {
        Ptr::new(!self.value(), self.tape_len())
    }
}

impl Debug for Ptr {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            f.debug_struct("Ptr")
                .field("value", &self.value())
                .field("tape_len", &self.tape_len())
                .finish()
        } else {
            Debug::fmt(&self.value(), f)
        }
    }
}

impl Display for Ptr {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.value(), f)
    }
}

impl LowerHex for Ptr {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        LowerHex::fmt(&self.value(), f)
    }
}

impl UpperHex for Ptr {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        UpperHex::fmt(&self.value(), f)
    }
}

#[cfg(test)]
mod tests {
    use super::{Cell, Ptr};

    #[test]
    fn ptr_into_cell() {
        let tape_len = 400;

        let zero = Ptr::new(0, tape_len);
        assert_eq!(zero.into_cell(), Cell::new(0));

        let two_fifty_five = Ptr::new(255, tape_len);
        assert_eq!(two_fifty_five.into_cell(), Cell::new(255));

        let two_fifty_six = Ptr::new(256, tape_len);
        assert_eq!(two_fifty_six.into_cell(), Cell::new(0));
    }

    #[test]
    fn ptr_tape_wrapping() {
        let tape_len = 400;

        let over_one = Ptr::new(tape_len, tape_len);
        assert_eq!(over_one.value(), 0);

        let underflow = Ptr::new(0, tape_len).wrapping_sub(Ptr::new(1, tape_len));
        assert_eq!(underflow.value(), tape_len - 1);

        let overflow_add = Ptr::new(10, tape_len)
            .wrapping_add(Ptr::new(tape_len / 2, tape_len))
            .wrapping_add(Ptr::new(tape_len / 2, tape_len));
        assert_eq!(overflow_add.value(), 10);

        let overflow_sub = Ptr::new(10, tape_len)
            .wrapping_sub(Ptr::new(tape_len / 2, tape_len))
            .wrapping_sub(Ptr::new(tape_len / 2, tape_len));
        assert_eq!(overflow_sub.value(), 10);

        let overflow_new = Ptr::new(tape_len + 50, tape_len);
        assert_eq!(overflow_new.value(), 50);
        assert_eq!(
            overflow_new,
            Ptr::new(tape_len, tape_len).wrapping_add(Ptr::new(50, tape_len)),
        );
    }

    #[test]
    fn long_tape_lengths() {
        let tape_len = u16::MAX;

        let over_one = Ptr::new(tape_len, tape_len);
        assert_eq!(over_one.value(), 0);

        let underflow = Ptr::new(0, tape_len).wrapping_sub(Ptr::new(1, tape_len));
        assert_eq!(underflow.value(), tape_len - 1);

        let overflow_add = Ptr::new(10, tape_len)
            .wrapping_add(Ptr::new(tape_len / 2, tape_len))
            .wrapping_add(Ptr::new(tape_len / 2, tape_len));
        assert_eq!(overflow_add.value(), 10);

        let overflow_sub = Ptr::new(10, tape_len)
            .wrapping_sub(Ptr::new(tape_len / 2, tape_len))
            .wrapping_sub(Ptr::new(tape_len / 2, tape_len));
        assert_eq!(overflow_sub.value(), 10);

        let overflow_new = Ptr::new(tape_len + 50, tape_len);
        assert_eq!(overflow_new.value(), 50);
        assert_eq!(
            overflow_new,
            Ptr::new(tape_len, tape_len).wrapping_add(Ptr::new(50, tape_len)),
        );
    }
}
