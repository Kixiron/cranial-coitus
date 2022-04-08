use crate::passes::dataflow::domain::{
    bitmap_u16::{U16Bitmap, BITS_LEN},
    utils,
};
use std::{cmp::Ordering, fmt::Debug, iter::FusedIterator};

impl FromIterator<u16> for U16Bitmap {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = u16>,
    {
        let mut this = Self::empty();
        iter.into_iter().for_each(|value| {
            this.insert(value);
        });

        this
    }
}

impl Extend<u16> for U16Bitmap {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = u16>,
    {
        iter.into_iter().for_each(|value| {
            self.insert(value);
        });
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Iter<'a> {
    bitmap: &'a U16Bitmap,
    forward_value: u64,
    forward_index: usize,
    backward_value: u64,
    backward_index: usize,
    length: usize,
}

impl<'a> Iter<'a> {
    #[inline]
    pub(crate) fn new(bitmap: &'a U16Bitmap) -> Self {
        Self {
            bitmap,
            forward_value: unsafe { *bitmap.bits.get_unchecked(0) },
            forward_index: 0,
            backward_value: unsafe { *bitmap.bits.get_unchecked(BITS_LEN - 1) },
            backward_index: BITS_LEN - 1,
            length: bitmap.len(),
        }
    }
}

impl<'a> From<&'a U16Bitmap> for Iter<'a> {
    #[inline]
    fn from(bitmap: &'a U16Bitmap) -> Self {
        Self::new(bitmap)
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = u16;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(value) = utils::pop_lsb(&mut self.forward_value) {
                self.length = self.length.saturating_sub(1);
                let value = (self.forward_index * u64::BITS as usize) + value as usize;

                break Some(value as u16);
            } else {
                self.forward_index += 1;

                let cmp = self.forward_index.cmp(&self.backward_index);

                // Match arms can be reordered, this ordering is perf sensitive
                self.forward_value = if cmp == Ordering::Less {
                    unsafe { *self.bitmap.bits.get_unchecked(self.forward_index) }
                } else if cmp == Ordering::Equal {
                    self.backward_value
                } else {
                    break None;
                };
            }
        }
    }

    fn for_each<F>(mut self, mut for_each: F)
    where
        F: FnMut(Self::Item),
    {
        while let Some(lsb) = utils::pop_lsb(&mut self.forward_value) {
            for_each((lsb as usize + (self.forward_index * u64::BITS as usize)) as u16);
        }

        (self.forward_index + 1..self.backward_index).for_each(|index| {
            debug_assert!(index < BITS_LEN);
            let mut chunk = unsafe { *self.bitmap.bits.get_unchecked(index) };

            while let Some(lsb) = utils::pop_lsb(&mut chunk) {
                for_each((lsb as usize + (index * u64::BITS as usize)) as u16);
            }
        });

        while let Some(lsb) = utils::pop_lsb(&mut self.backward_value) {
            for_each((lsb as usize + (self.backward_index * u64::BITS as usize)) as u16);
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }

    #[inline]
    fn count(self) -> usize {
        self.length
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[inline]
    fn min(mut self) -> Option<Self::Item> {
        self.next()
    }

    #[inline]
    fn max(self) -> Option<Self::Item> {
        self.last()
    }
}

impl<'a> DoubleEndedIterator for Iter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let value = if self.backward_index < self.forward_index {
                &mut self.forward_value
            } else {
                &mut self.backward_value
            };

            if *value == 0 {
                if self.backward_index <= self.forward_index {
                    break None;
                } else {
                    self.backward_index -= 1;

                    debug_assert!(self.backward_index < BITS_LEN);
                    self.backward_value =
                        unsafe { *self.bitmap.bits.get_unchecked(self.backward_index) };
                }
            } else {
                self.length = self.length.saturating_sub(1);

                let index_from_left = value.leading_zeros() as usize;
                let index = 63 - index_from_left;
                *value &= !(1 << index);

                let value = (self.backward_index * u64::BITS as usize) + index;
                break Some(value as u16);
            }
        }
    }
}

impl<'a> ExactSizeIterator for Iter<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.length
    }
}

impl<'a> FusedIterator for Iter<'a> {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntoIter {
    bitmap: U16Bitmap,
    forward_index: usize,
    backward_index: usize,
    length: usize,
}

impl IntoIter {
    #[inline]
    pub(crate) fn new(bitmap: U16Bitmap) -> Self {
        let length = bitmap.len();

        Self {
            bitmap,
            forward_index: 0,
            backward_index: BITS_LEN - 1,
            length,
        }
    }
}

impl From<U16Bitmap> for IntoIter {
    #[inline]
    fn from(bitmap: U16Bitmap) -> Self {
        Self::new(bitmap)
    }
}

impl Iterator for IntoIter {
    type Item = u16;

    fn next(&mut self) -> Option<Self::Item> {
        for index in self.forward_index..self.backward_index + 1 {
            self.forward_index = index;

            debug_assert!(index < BITS_LEN);
            let chunk = unsafe { self.bitmap.bits.get_unchecked_mut(index) };

            if let Some(lsb) = utils::pop_lsb(chunk) {
                self.length = self.length.saturating_sub(1);
                return Some((lsb as usize + (index * u64::BITS as usize)) as u16);
            }
        }

        None
    }

    fn for_each<F>(mut self, mut each: F)
    where
        F: FnMut(Self::Item),
    {
        (self.forward_index..self.backward_index + 1).for_each(|index| {
            debug_assert!(index < BITS_LEN);
            let chunk = unsafe { self.bitmap.bits.get_unchecked_mut(index) };

            while let Some(lsb) = utils::pop_lsb(chunk) {
                each((lsb as usize + (index * u64::BITS as usize)) as u16);
            }
        });
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }

    #[inline]
    fn count(self) -> usize {
        self.length
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[inline]
    fn min(mut self) -> Option<Self::Item> {
        self.next()
    }

    #[inline]
    fn max(self) -> Option<Self::Item> {
        self.last()
    }
}

impl DoubleEndedIterator for IntoIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        // `Range` (`a..b`) is faster than `InclusiveRange` (`a..=b`).
        for index in (self.forward_index..(self.backward_index + 1)).rev() {
            self.backward_index = index;

            // SAFETY: This invariant is tested.
            debug_assert!(index < BITS_LEN);
            let chunk = unsafe { self.bitmap.bits.get_unchecked_mut(index) };

            if let Some(msb) = utils::pop_msb(chunk) {
                self.length = self.length.saturating_sub(1);
                return Some((msb as usize + (index * u64::BITS as usize)) as u16);
            }
        }

        None
    }
}

impl ExactSizeIterator for IntoIter {
    #[inline]
    fn len(&self) -> usize {
        self.length
    }
}

impl FusedIterator for IntoIter {}

#[cfg(test)]
mod tests {
    use crate::passes::dataflow::domain::bitmap_u16::U16Bitmap;

    const VALUES: &[u16] = &[0, 5, 100, 10000, u16::MAX / 2, u16::MAX];

    #[test]
    fn iter() {
        let bitmap: Vec<_> = VALUES
            .iter()
            .copied()
            .collect::<U16Bitmap>()
            .iter()
            .collect();
        assert_eq!(&bitmap, VALUES);
    }

    #[test]
    fn into_iter() {
        let bitmap: Vec<_> = VALUES
            .iter()
            .copied()
            .collect::<U16Bitmap>()
            .into_iter()
            .collect();
        assert_eq!(&bitmap, VALUES);
    }

    #[test]
    fn iter_rev() {
        let bitmap: Vec<_> = VALUES
            .iter()
            .copied()
            .collect::<U16Bitmap>()
            .iter()
            .rev()
            .collect();
        let expected = VALUES.iter().copied().rev().collect::<Vec<_>>();

        assert_eq!(bitmap, expected);
    }

    #[test]
    fn into_iter_rev() {
        let bitmap: Vec<_> = VALUES
            .iter()
            .copied()
            .collect::<U16Bitmap>()
            .into_iter()
            .rev()
            .collect();
        let expected = VALUES.iter().copied().rev().collect::<Vec<_>>();

        assert_eq!(bitmap, expected);
    }

    #[test]
    fn iter_front_back() {
        let bitmap: U16Bitmap = VALUES.iter().copied().collect();
        let mut iter = bitmap.iter();

        assert_eq!(iter.next(), Some(VALUES[0]));
        debug_assert_eq!(iter.len(), VALUES.len() - 1);

        let (mut idx, mut done) = (VALUES.len() - 1, false);
        while let Some(returned) = iter.next_back() {
            debug_assert!(!done);

            assert_eq!(returned, VALUES[idx]);
            idx = idx.checked_sub(1).unwrap_or_else(|| {
                done = true;
                0
            });
        }
    }

    #[test]
    fn into_iter_front_back() {
        let bitmap: U16Bitmap = VALUES.iter().copied().collect();
        let mut iter = bitmap.into_iter();

        assert_eq!(iter.next(), Some(VALUES[0]));

        let mut idx = VALUES.len() - 1;
        while let Some(returned) = iter.next_back() {
            assert_eq!(returned, VALUES[idx]);
            idx -= 1;
        }
    }
}
