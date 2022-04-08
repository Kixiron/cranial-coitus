mod arch;
mod iter;

pub(super) use iter::{IntoIter, Iter};

use crate::passes::dataflow::domain::ByteSet;
use std::{
    cell::{Cell, RefCell},
    fmt::{self, Debug},
    mem::{ManuallyDrop, MaybeUninit},
    thread,
};

const BITS_LEN: usize = 1024;
const MAX_LEN: u32 = u16::MAX as u32 + 1;

type BitArray = [u64; BITS_LEN];
type BitBox = ManuallyDrop<Box<BitArray>>;

thread_local! {
    static BITMAP_CACHE: RefCell<Vec<Box<BitArray>>> = RefCell::new(Vec::new());
}

pub struct U16Bitmap {
    bits: BitBox,
    length: Cell<Option<u32>>,
}

impl U16Bitmap {
    pub fn empty() -> Self {
        Self {
            bits: zeroed_bit_box(),
            length: Cell::new(Some(0)),
        }
    }

    pub fn full() -> Self {
        let mut bits = uninit_bit_box();
        arch::set_uninit_bits_one(&mut bits);

        Self {
            // Safety: All bits have been initialized
            bits: unsafe { assume_bit_box_init(bits) },
            length: Cell::new(Some(MAX_LEN)),
        }
    }

    pub fn singleton(value: u16) -> Self {
        let mut this = Self::empty();
        this.insert(value);
        this
    }

    pub fn is_empty(&self) -> bool {
        if matches!(self.raw_len(), Some(0)) {
            true
        } else {
            let is_empty = arch::all_bits_are_unset(self.as_array());
            if is_empty {
                self.length.set(Some(0));
            }

            is_empty
        }
    }

    pub fn is_full(&self) -> bool {
        if matches!(self.raw_len(), Some(MAX_LEN)) {
            true
        } else {
            let is_full = arch::all_bits_are_set(self.as_array());
            if is_full {
                self.length.set(Some(MAX_LEN));
            }

            is_full
        }
    }

    pub fn len(&self) -> usize {
        let length = self
            .length
            .get()
            .unwrap_or_else(|| arch::popcount(self.as_array()));
        self.length.set(Some(length));

        length as usize
    }

    fn raw_len(&self) -> Option<u32> {
        self.length.get()
    }

    pub fn contains(&self, value: u16) -> bool {
        let (index, offset) = (key(value), bit(value));
        debug_assert!(index < BITS_LEN);

        let lane = unsafe { *self.bits.get_unchecked(index) };
        lane & (1 << offset) != 0
    }

    pub fn insert(&mut self, value: u16) -> bool {
        let (index, offset) = (key(value), bit(value));
        debug_assert!(index < BITS_LEN);

        let target = unsafe { self.bits.get_unchecked_mut(index) };
        let old = *target;
        *target |= 1 << offset;

        let was_inserted = *target != old;
        if was_inserted {
            self.length.update(|length| length.map(|length| length + 1));
        }

        was_inserted
    }

    pub fn remove(&mut self, value: u16) -> bool {
        let (index, offset) = (key(value), bit(value));
        debug_assert!(index < BITS_LEN);

        let target = unsafe { self.bits.get_unchecked_mut(index) };
        let old = *target;
        *target &= !(1 << offset);

        let was_removed = *target != old;
        if was_removed {
            self.length.update(|length| length.map(|length| length - 1));
        }

        was_removed
    }

    pub fn as_singleton(&self) -> Option<u16> {
        if self.len() != 1 {
            None
        } else {
            self.bits.iter().enumerate().find_map(|(idx, &bits)| {
                (bits != 0).then(|| (idx as u16 * u64::BITS as u16) + bits.trailing_zeros() as u16)
            })
        }
    }

    pub fn fill(&mut self) {
        self.length.set(Some(MAX_LEN));
        arch::set_bits_one(self.as_mut_array());
    }

    pub fn clear(&mut self) {
        self.length.set(Some(0));
        arch::set_bits_zero(self.as_mut_array());
    }

    /// Fills `self` with the set of all integers contained
    /// in `self`, `other` or both `self` and `other
    ///
    /// Computes `self ∪ other`, storing the result in `self` and returning
    /// a boolean as to whether or not `self` has changed any
    ///
    pub fn union(&mut self, other: &Self) -> bool {
        self.length.set(None);
        arch::union(self.as_mut_array(), other.as_array())
    }

    /// Fills `output` with the set of all integers contained
    /// in `self`, `other` or both `self` and `other
    ///
    /// Computes `self ∪ other`, storing the result in `output` and returning
    /// a boolean as to whether or not `output` has changed any in comparison
    /// to `self` (that is, this function returns `true` if `output != self`)
    ///
    pub fn union_into(&self, other: &Self, output: &mut Self) -> bool {
        output.length.set(None);
        arch::union_into(self.as_array(), other.as_array(), output.as_mut_array())
    }

    /// Creates a new [`U16Bitmap`] with the set of all integers contained
    /// in `self`, `other` or both `self` and `other
    ///
    /// Computes `self ∪ other`, returning the result.
    ///
    pub fn union_new(&self, other: &Self) -> Self {
        let mut uninit = uninit_bit_box();
        arch::union_into_uninit(self.as_array(), other.as_array(), &mut *uninit);

        Self {
            bits: unsafe { assume_bit_box_init(uninit) },
            length: Cell::new(None),
        }
    }

    pub(super) fn union_bytes(&mut self, rhs: ByteSet) -> bool {
        let array_ref = (&mut self.bits[..ByteSet::LEN]).try_into().unwrap();
        let lhs = ByteSet::from_mut(array_ref);
        lhs.union(rhs)
    }

    pub fn intersect(&mut self, other: &Self) {
        self.length.set(None);
        arch::intersect(self.as_mut_array(), other.as_array())
    }

    pub fn intersect_into(&self, other: &Self, output: &mut Self) {
        output.length.set(None);
        arch::intersect_into(self.as_array(), other.as_array(), output.as_mut_array())
    }

    pub fn intersect_new(&self, other: &Self) -> Self {
        let mut uninit = uninit_bit_box();
        arch::intersect_into_uninit(self.as_array(), other.as_array(), &mut *uninit);

        Self {
            bits: unsafe { assume_bit_box_init(uninit) },
            length: Cell::new(None),
        }
    }

    pub(super) fn intersect_bytes(&mut self, rhs: &ByteSet) -> bool {
        let array_ref = (&mut self.bits[..ByteSet::LEN]).try_into().unwrap();
        let lhs = ByteSet::from_mut(array_ref);
        lhs.intersect(rhs)
    }

    pub(super) fn intersect_bytes_new(&self, rhs: &ByteSet) -> Self {
        let mut lhs = self.clone();
        lhs.intersect_bytes(rhs);
        lhs
    }

    pub fn intersects(&self, other: &Self) -> bool {
        arch::intersects(self.as_array(), other.as_array())
    }

    pub(crate) fn intersects_bytes(&self, other: ByteSet) -> bool {
        let lhs = ByteSet::from_ref((&self.bits[..ByteSet::LEN]).try_into().unwrap());
        lhs.intersects(other)
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        arch::is_disjoint(self.as_array(), other.as_array())
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        arch::is_subset(self.as_array(), other.as_array())
    }

    #[inline]
    fn as_array(&self) -> &BitArray {
        &*self.bits
    }

    #[inline]
    fn as_mut_array(&mut self) -> &mut BitArray {
        &mut *self.bits
    }

    pub(super) fn iter(&self) -> Iter<'_> {
        Iter::new(self)
    }

    pub(super) fn into_iter(self) -> IntoIter {
        IntoIter::new(self)
    }
}

impl Default for U16Bitmap {
    fn default() -> Self {
        Self::empty()
    }
}

impl PartialEq for U16Bitmap {
    fn eq(&self, other: &Self) -> bool {
        // If the lengths of the two bitmaps aren't equal, their contents
        // can't possibly be. This allows us to potentially skip on the more
        // expensive equality check if both bitmaps already know their
        // lengths
        let equal_lengths = self
            .raw_len()
            .zip(other.raw_len())
            .map_or(true, |(lhs, rhs)| lhs == rhs);

        equal_lengths && arch::bitmap_eq(self.as_array(), other.as_array())
    }
}

impl Eq for U16Bitmap {}

impl Debug for U16Bitmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl Clone for U16Bitmap {
    fn clone(&self) -> Self {
        let mut bits = uninit_bit_box();
        MaybeUninit::write_slice(&mut *bits, self.as_array());

        let clone = Self {
            // Safety: We've initialized the bits
            bits: unsafe { assume_bit_box_init(bits) },
            length: self.length.clone(),
        };
        debug_assert_eq!(self, &clone);

        clone
    }

    fn clone_from(&mut self, other: &Self) {
        self.as_mut_array().copy_from_slice(other.as_array());
        self.length = other.length.clone();
        debug_assert_eq!(self, other);
    }
}

impl Drop for U16Bitmap {
    fn drop(&mut self) {
        // If the thread is panicking we just want to deallocate the box
        if thread::panicking() {
            // Safety: The box's contents will never be touched again
            unsafe { ManuallyDrop::drop(&mut self.bits) };

        // Otherwise we want to keep the allocation around for reuse
        } else {
            // Safety: The bits of this bitmap won't be touched again
            let bits = unsafe { ManuallyDrop::take(&mut self.bits) };
            BITMAP_CACHE.with_borrow_mut(|cache| cache.push(bits));
        }
    }
}

#[inline(always)]
fn key(index: u16) -> usize {
    index as usize / 64
}

#[inline(always)]
fn bit(index: u16) -> usize {
    index as usize % 64
}

#[inline]
fn pop_bit_cache() -> Option<BitBox> {
    BITMAP_CACHE.with_borrow_mut(|cache| cache.pop().map(ManuallyDrop::new))
}

#[inline]
fn zeroed_bit_box() -> BitBox {
    pop_bit_cache()
        .map(|mut bits| {
            arch::set_bits_zero(&mut bits);
            bits
        })
        // Safety: u64s are zeroable
        .unwrap_or_else(|| ManuallyDrop::new(unsafe { Box::new_zeroed().assume_init() }))
}

#[inline]
fn uninit_bit_box() -> Box<[MaybeUninit<u64>; BITS_LEN]> {
    pop_bit_cache()
        .map(|bits| {
            let bits = ManuallyDrop::into_inner(bits);
            // Safety: We can act like this array isn't initialized
            unsafe { Box::from_raw(Box::into_raw(bits).cast::<[MaybeUninit<u64>; BITS_LEN]>()) }
        })
        // Safety: We're creating an uninit array
        .unwrap_or_else(|| unsafe { Box::new_uninit().assume_init() })
}

/// # Safety
///
/// The caller must ensure that all bits within the bit box are initialized
#[inline]
unsafe fn assume_bit_box_init(bits: Box<[MaybeUninit<u64>; BITS_LEN]>) -> BitBox {
    ManuallyDrop::new(unsafe { Box::from_raw(Box::into_raw(bits).cast::<BitArray>()) })
}

#[cfg(test)]
mod tests {
    use crate::passes::dataflow::domain::bitmap_u16::U16Bitmap;

    #[test]
    fn bitmap_as_singleton() {
        let mut bitmap = U16Bitmap::singleton(10);
        assert_eq!(bitmap.as_singleton(), Some(10));

        bitmap.insert(200);
        assert_eq!(bitmap.as_singleton(), None);

        bitmap.clear();
        bitmap.insert(20_000);
        assert_eq!(bitmap.as_singleton(), Some(20_000));
    }

    #[test]
    fn drop_a_bunch() {
        let mut maps = Vec::new();
        for _ in 0..100_000 {
            maps.push(U16Bitmap::empty());
        }
    }

    #[test]
    fn equality() {
        let mut empty = U16Bitmap::empty();
        let full = U16Bitmap::full();

        assert_ne!(empty, full);

        // Force an actual equality check instead of just checking
        // for length equality
        full.length.set(None);
        assert_ne!(empty, full);

        empty.insert(10000);
        empty.insert(u16::MAX);
        assert_ne!(empty, full);

        assert_eq!(full, full);
        assert_eq!(empty, empty);
    }

    #[test]
    fn union() {
        let mut empty = U16Bitmap::empty();
        let full = U16Bitmap::full();

        assert_eq!(empty.union_new(&full), full);

        let mut half: U16Bitmap = (0..=u16::MAX).step_by(2).collect();
        assert_eq!(half.union_new(&empty), half);
        assert_eq!(half.union_new(&half), half);

        assert!(!half.union(&empty));
        assert!(empty.union(&half));
        assert_eq!(empty, half);
    }
}
