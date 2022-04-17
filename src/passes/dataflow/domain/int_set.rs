use crate::{
    passes::dataflow::domain::{
        bitmap_u16::{self, U16Bitmap},
        ByteSet,
    },
    utils::DebugCollapseRanges,
    values::Ptr,
};
use std::{
    fmt::{self, Debug, Display, Write},
    iter::FusedIterator,
};

#[derive(Clone, PartialEq, Default)]
pub struct IntSet {
    bitmap: U16Bitmap,
    pub(super) tape_len: u16,
}

impl IntSet {
    pub fn empty(tape_len: u16) -> Self {
        Self {
            bitmap: U16Bitmap::empty(),
            tape_len,
        }
    }

    pub fn full(tape_len: u16) -> Self {
        Self {
            bitmap: U16Bitmap::full(),
            tape_len,
        }
    }

    pub fn singleton(value: Ptr) -> Self {
        // let mut this = Self::empty(value.tape_len());
        // this.values.insert(value.value() as u32);
        // this
        Self {
            bitmap: U16Bitmap::singleton(value.value()),
            tape_len: value.tape_len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.bitmap.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.bitmap.is_full()
    }

    pub fn len(&self) -> usize {
        self.bitmap.len()
    }

    pub fn add(&mut self, value: Ptr) {
        debug_assert_eq!(self.tape_len, value.tape_len());
        self.bitmap.insert(value.value());
    }

    pub fn union(&mut self, other: &Self) -> bool {
        debug_assert_eq!(self.tape_len, other.tape_len);

        // self.values &= &other.values;
        self.bitmap.union(&other.bitmap)
    }

    pub(crate) fn union_bytes(&mut self, rhs: ByteSet) -> bool {
        self.bitmap.union_bytes(rhs)
    }

    pub fn as_singleton(&self) -> Option<Ptr> {
        // (self.values.len() == 1).then(|| self.iter().next().unwrap())
        self.bitmap
            .as_singleton()
            .map(|value| Ptr::new(value, self.tape_len))
    }

    pub fn intersects(&self, other: &Self) -> bool {
        debug_assert_eq!(self.tape_len, other.tape_len);
        self.bitmap.intersects(&other.bitmap)
    }

    pub(crate) fn intersects_bytes(&self, other: ByteSet) -> bool {
        self.bitmap.intersects_bytes(other)
    }

    pub fn intersect(&self, other: &Self) -> Self {
        debug_assert_eq!(self.tape_len, other.tape_len);

        Self {
            // values: &self.values | &other.values,
            bitmap: self.bitmap.intersect_new(&other.bitmap),
            tape_len: self.tape_len,
        }
    }

    pub fn intersect_bytes(&self, other: &ByteSet) -> Self {
        Self {
            bitmap: self.bitmap.intersect_bytes_new(other),
            tape_len: self.tape_len,
        }
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        debug_assert_eq!(self.tape_len, other.tape_len);
        self.bitmap.is_subset(&other.bitmap)
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        debug_assert_eq!(self.tape_len, other.tape_len);
        self.bitmap.is_disjoint(&other.bitmap)
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_> {
        Iter::new(self)
    }
}

impl From<Ptr> for IntSet {
    fn from(int: Ptr) -> Self {
        Self::singleton(int)
    }
}

impl Debug for IntSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            f.write_char('Ø')
        } else if self.is_full() {
            write!(f, "[0, {}]", u16::MAX)
        } else {
            f.debug_set()
                .entries(DebugCollapseRanges::new(self.iter().map(Ptr::value)))
                .finish()
        }
    }
}

impl Display for IntSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

impl<'a> IntoIterator for &'a IntSet {
    type IntoIter = Iter<'a>;
    type Item = Ptr;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

impl IntoIterator for IntSet {
    type IntoIter = IntoIter;
    type Item = Ptr;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Iter<'a> {
    iter: bitmap_u16::Iter<'a>,
    tape_len: u16,
}

impl<'a> Iter<'a> {
    #[inline]
    pub fn new(set: &'a IntSet) -> Self {
        Self {
            iter: set.bitmap.iter(),
            tape_len: set.tape_len,
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = Ptr;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|value| Ptr::new(value, self.tape_len))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn for_each<F>(self, mut for_each: F)
    where
        F: FnMut(Self::Item),
    {
        self.iter
            .for_each(|value| for_each(Ptr::new(value, self.tape_len)));
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.iter.last().map(|value| Ptr::new(value, self.tape_len))
    }

    #[inline]
    fn min(self) -> Option<Self::Item> {
        self.iter.min().map(|value| Ptr::new(value, self.tape_len))
    }

    #[inline]
    fn max(self) -> Option<Self::Item> {
        self.iter.max().map(|value| Ptr::new(value, self.tape_len))
    }
}

impl<'a> DoubleEndedIterator for Iter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .next_back()
            .map(|value| Ptr::new(value, self.tape_len))
    }
}

impl<'a> ExactSizeIterator for Iter<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a> FusedIterator for Iter<'a> {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntoIter {
    iter: bitmap_u16::IntoIter,
    tape_len: u16,
}

impl IntoIter {
    pub fn new(set: IntSet) -> Self {
        Self {
            iter: set.bitmap.into_iter(),
            tape_len: set.tape_len,
        }
    }
}

impl Iterator for IntoIter {
    type Item = Ptr;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|value| Ptr::new(value, self.tape_len))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn for_each<F>(self, mut for_each: F)
    where
        F: FnMut(Self::Item),
    {
        self.iter
            .for_each(|value| for_each(Ptr::new(value, self.tape_len)));
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.iter.last().map(|value| Ptr::new(value, self.tape_len))
    }

    #[inline]
    fn min(self) -> Option<Self::Item> {
        self.iter.min().map(|value| Ptr::new(value, self.tape_len))
    }

    #[inline]
    fn max(self) -> Option<Self::Item> {
        self.iter.max().map(|value| Ptr::new(value, self.tape_len))
    }
}

impl DoubleEndedIterator for IntoIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .next_back()
            .map(|value| Ptr::new(value, self.tape_len))
    }
}

impl ExactSizeIterator for IntoIter {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}
