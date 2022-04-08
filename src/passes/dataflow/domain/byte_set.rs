use crate::{
    passes::dataflow::domain::{utils as bit_utils, IntSet},
    utils::{self, DebugCollapseRanges},
    values::{Cell, Ptr},
};
use std::{
    cell::RefCell,
    fmt::{self, Debug, Display, Write},
    intrinsics::transmute,
    iter::FusedIterator,
    mem::{size_of, take, MaybeUninit},
    ops::{Index, IndexMut},
    simd::{mask64x4, u64x4},
    slice, thread,
};

thread_local! {
    static TAPE_CACHE: RefCell<Vec<Box<[ByteSet]>>> = RefCell::new(Vec::new());
}

#[repr(transparent)]
pub struct ProgramTape {
    tape: Box<[ByteSet]>,
}

impl ProgramTape {
    pub fn zeroed(tape_len: u16) -> Self {
        // Create a completely empty program tape
        let mut this = TAPE_CACHE
            .with_borrow_mut(|cache| cache.pop())
            .map(|tape| {
                let mut tape = Self { tape };
                tape.clear();
                tape
            })
            .unwrap_or_else(|| {
                let tape = unsafe { Box::new_zeroed_slice(tape_len as usize).assume_init() };
                Self { tape }
            });

        // Set every cell's value to zero
        this.as_mut_arrays().fill(ByteSet::singleton(0).values);
        this
    }

    pub const fn len(&self) -> usize {
        self.tape.len()
    }

    pub fn clear(&mut self) {
        self.as_mut_bytes().fill(0);
    }

    pub fn iter_mut(&mut self) -> slice::IterMut<'_, ByteSet> {
        self.tape.iter_mut()
    }

    pub fn fill(&mut self) {
        self.as_mut_arrays().fill(ByteSet::full().values);
    }

    pub fn union(&mut self, other: &Self) -> bool {
        assert_eq!(self.len(), other.len());
        let (lhs, rhs) = (self.as_mut_arrays(), other.as_arrays());

        let mut did_change = mask64x4::splat(false);
        lhs.iter_mut().zip(rhs).for_each(|(lhs, &rhs)| {
            let old_lhs = u64x4::from_array(*lhs);
            *lhs = *(old_lhs | u64x4::from_array(rhs)).as_array();
            did_change |= u64x4::from_array(*lhs).lanes_ne(old_lhs);
        });

        did_change.any()
    }

    pub fn union_into(&mut self, other: &Self, output: &mut Self) -> bool {
        assert!(self.len() == other.len() && self.len() == output.len());
        let (lhs, rhs, output) = (self.as_arrays(), other.as_arrays(), output.as_mut_arrays());

        let mut did_change = mask64x4::splat(false);
        output
            .iter_mut()
            .zip(lhs)
            .zip(rhs)
            .for_each(|((output, &lhs), &rhs)| {
                let (lhs, rhs) = (u64x4::from_array(lhs), u64x4::from_array(rhs));
                let union = lhs | rhs;
                did_change |= union.lanes_ne(lhs);
                *output = *union.as_array();
            });

        did_change.any()
    }

    fn as_ptr(&self) -> *const ByteSet {
        self.tape.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut ByteSet {
        self.tape.as_mut_ptr()
    }

    fn as_arrays(&self) -> &[[u64; SLOTS]] {
        let (ptr, length) = (self.as_ptr().cast(), self.len());
        unsafe { slice::from_raw_parts(ptr, length) }
    }

    fn as_mut_arrays(&mut self) -> &mut [[u64; SLOTS]] {
        let (ptr, length) = (self.as_mut_ptr().cast(), self.len());
        unsafe { slice::from_raw_parts_mut(ptr, length) }
    }

    fn as_mut_bytes(&mut self) -> &mut [u8] {
        let (ptr, len) = (
            self.as_mut_ptr().cast(),
            self.len() * SLOTS * size_of::<u64>(),
        );
        unsafe { slice::from_raw_parts_mut(ptr, len) }
    }
}

impl Index<Ptr> for ProgramTape {
    type Output = ByteSet;

    fn index(&self, idx: Ptr) -> &Self::Output {
        debug_assert_eq!(self.tape.len(), idx.tape_len() as usize);
        debug_assert!((idx.value() as usize) < self.tape.len());

        unsafe { self.tape.get_unchecked(idx.value() as usize) }
    }
}

impl IndexMut<Ptr> for ProgramTape {
    fn index_mut(&mut self, idx: Ptr) -> &mut Self::Output {
        debug_assert_eq!(self.tape.len(), idx.tape_len() as usize);
        debug_assert!((idx.value() as usize) < self.tape.len());

        unsafe { self.tape.get_unchecked_mut(idx.value() as usize) }
    }
}

impl Debug for ProgramTape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(utils::debug_collapse(&self.tape))
            .finish()
    }
}

impl Clone for ProgramTape {
    fn clone(&self) -> Self {
        if let Some(mut tape) = TAPE_CACHE.with_borrow_mut(|cache| cache.pop()) {
            tape.copy_from_slice(&self.tape);
            Self { tape }
        } else {
            let mut tape = Box::new_uninit_slice(self.len());
            MaybeUninit::write_slice(&mut tape, &self.tape);

            Self {
                tape: unsafe { tape.assume_init() },
            }
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.tape.copy_from_slice(&source.tape);
    }
}

impl Drop for ProgramTape {
    fn drop(&mut self) {
        if !thread::panicking() {
            TAPE_CACHE.with_borrow_mut(|cache| cache.push(take(&mut self.tape)));
        }
    }
}

const SLOTS: usize = 4;
const LAST_SLOT_IDX: usize = SLOTS - 1;

#[inline]
const fn chunk_index_and_offset(byte: u8) -> (usize, usize) {
    let byte = byte as usize;
    let index = byte >> 6;
    let offset = byte & 0b0011_1111;

    (index, offset)
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct ByteSet {
    values: [u64; SLOTS],
}

impl ByteSet {
    pub const LEN: usize = SLOTS;

    pub const fn empty() -> Self {
        Self { values: [0; SLOTS] }
    }

    pub fn full() -> Self {
        Self {
            values: [u64::MAX; SLOTS],
        }
    }

    pub fn singleton(byte: u8) -> Self {
        let mut this = Self::empty();
        this.insert(byte);
        this
    }

    pub fn from_ref(values: &[u64; SLOTS]) -> &Self {
        // Safety: ByteSet is transparent over an array
        unsafe { transmute(values) }
    }

    pub fn from_mut(values: &mut [u64; SLOTS]) -> &mut Self {
        // Safety: ByteSet is transparent over an array
        unsafe { transmute(values) }
    }

    pub fn clear(&mut self) {
        self.values = [0; SLOTS];
    }

    pub fn fill(&mut self) {
        *self = Self::full();
    }

    pub fn len(&self) -> usize {
        self.values
            .iter()
            .map(|&bits| bits.count_ones())
            .sum::<u32>() as usize
    }

    pub fn is_empty(&self) -> bool {
        *self == Self::empty()
    }

    pub fn is_full(&self) -> bool {
        *self == Self::full()
    }

    pub fn insert(&mut self, byte: u8) -> bool {
        let (index, offset) = chunk_index_and_offset(byte);
        debug_assert!(index < self.values.len());

        let target = unsafe { self.values.get_unchecked_mut(index) };
        let old = *target;
        *target |= 1 << offset;

        *target != old
    }

    #[allow(dead_code)]
    pub fn remove(&mut self, byte: u8) -> bool {
        let (index, offset) = chunk_index_and_offset(byte);
        debug_assert!(index < self.values.len());

        let target = unsafe { self.values.get_unchecked_mut(index) };
        let old = *target;
        *target &= !(1 << offset);

        *target != old
    }

    pub fn union(&mut self, other: Self) -> bool {
        let (lhs, rhs) = (self.to_simd(), other.to_simd());

        let old = self.values;
        self.values = *(lhs | rhs).as_array();

        self.values != old
    }

    pub fn intersect(&mut self, other: &Self) -> bool {
        let (lhs, rhs) = (self.to_simd(), other.to_simd());

        let old = self.values;
        self.values = *(lhs & rhs).as_array();

        self.values != old
    }

    pub fn intersection(&self, other: &Self) -> Self {
        let (lhs, rhs) = (self.to_simd(), other.to_simd());

        Self {
            values: *(lhs & rhs).as_array(),
        }
    }

    pub fn intersects(&self, other: Self) -> bool {
        let (lhs, rhs) = (self.to_simd(), other.to_simd());
        lhs & rhs != u64x4::splat(0)
    }

    pub fn as_singleton(&self) -> Option<u8> {
        (self.len() == 1).then(|| self.iter().next().unwrap())
    }

    pub const fn iter(&self) -> Iter {
        Iter::new(*self)
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

    const fn to_simd(self) -> u64x4 {
        u64x4::from_array(self.values)
    }
}

impl IntoIterator for ByteSet {
    type IntoIter = Iter;
    type Item = u8;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
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
        if self.is_empty() {
            f.write_char('Ø')
        } else if self.is_full() {
            f.write_str("[0, 255]")
        } else {
            let elements = self.iter().map(|byte| byte as u16).collect::<Vec<_>>();
            f.debug_set()
                .entries(DebugCollapseRanges::new(&elements))
                .finish()
        }
    }
}

impl Display for ByteSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elements = self.iter().map(|byte| byte as u16).collect::<Vec<_>>();
        f.debug_set()
            .entries(DebugCollapseRanges::new(&elements))
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Iter {
    /// The set being iterated over. It is mutated in-place as bits are popped
    /// from each chunk.
    byte_set: ByteSet,

    /// The current chunk index when iterating forwards.
    forward_index: usize,

    /// The current chunk index when iterating backwards.
    backward_index: usize,
}

impl Iter {
    #[inline]
    pub(crate) const fn new(byte_set: ByteSet) -> Self {
        Self {
            byte_set,
            forward_index: 0,
            backward_index: LAST_SLOT_IDX,
        }
    }
}

impl From<ByteSet> for Iter {
    #[inline]
    fn from(byte_set: ByteSet) -> Self {
        Self::new(byte_set)
    }
}

impl Iterator for Iter {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        for index in self.forward_index..SLOTS {
            self.forward_index = index;

            debug_assert!(index < self.byte_set.values.len());
            let chunk = unsafe { self.byte_set.values.get_unchecked_mut(index) };

            if let Some(lsb) = bit_utils::pop_lsb(chunk) {
                return Some(lsb + (index * u64::BITS as usize) as u8);
            }
        }

        None
    }

    fn for_each<F>(mut self, mut each: F)
    where
        F: FnMut(u8),
    {
        (0..SLOTS).for_each(|index| {
            debug_assert!(index < self.byte_set.values.len());
            let chunk = unsafe { self.byte_set.values.get_unchecked_mut(index) };

            while let Some(lsb) = bit_utils::pop_lsb(chunk) {
                each(lsb + (index * u64::BITS as usize) as u8);
            }
        });
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(mut self) -> Option<u8> {
        self.next_back()
    }

    #[inline]
    fn min(mut self) -> Option<u8> {
        self.next()
    }

    #[inline]
    fn max(self) -> Option<u8> {
        self.last()
    }
}

impl DoubleEndedIterator for Iter {
    fn next_back(&mut self) -> Option<u8> {
        // `Range` (`a..b`) is faster than `InclusiveRange` (`a..=b`).
        for index in (0..(self.backward_index + 1)).rev() {
            self.backward_index = index;

            debug_assert!(index < self.byte_set.values.len());
            let chunk = unsafe { self.byte_set.values.get_unchecked_mut(index) };

            if let Some(msb) = bit_utils::pop_msb(chunk) {
                return Some(msb + (index * u64::BITS as usize) as u8);
            }
        }

        None
    }
}

impl ExactSizeIterator for Iter {
    #[inline]
    fn len(&self) -> usize {
        self.byte_set.len()
    }
}

// `ByteIter` does not produce more values after `None` is reached.
impl FusedIterator for Iter {}
