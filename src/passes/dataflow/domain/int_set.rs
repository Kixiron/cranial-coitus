use crate::{utils::DebugCollapseRanges, values::Ptr};
use roaring::RoaringBitmap;
use std::{
    fmt::{self, Debug, Display},
    mem::take,
};

#[derive(Clone, PartialEq, Default)]
pub struct IntSet {
    values: RoaringBitmap,
    pub(super) tape_len: u16,
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

    pub fn singleton(value: Ptr) -> Self {
        let mut this = Self::empty(value.tape_len());
        this.values.insert(value.value() as u32);
        this
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn len(&self) -> u64 {
        self.values.len()
    }

    pub fn add(&mut self, value: Ptr) {
        debug_assert_eq!(self.tape_len, value.tape_len());
        self.values.insert(value.value() as u32);
    }

    pub(super) fn union(&mut self, other: &Self) {
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
        let elements = self.iter().map(Ptr::value).collect::<Vec<_>>();
        f.debug_set()
            .entries(DebugCollapseRanges::new(&elements))
            .finish()
    }
}

impl Display for IntSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elements = self.iter().map(Ptr::value).collect::<Vec<_>>();
        f.debug_set()
            .entries(DebugCollapseRanges::new(&elements))
            .finish()
    }
}

#[allow(dead_code)]
mod bitmap_u16 {
    use std::{
        cell::Cell,
        mem::{transmute, MaybeUninit},
    };

    const BITS_LEN: usize = 1024; // u16::MAX as usize / u64::BITS as usize;

    pub struct U16Bitmap {
        bits: Box<[u64; BITS_LEN]>,
        length: Cell<Option<u16>>,
    }

    impl U16Bitmap {
        pub fn empty() -> Self {
            let bits = unsafe { Box::new_zeroed().assume_init() };
            Self {
                bits,
                length: Cell::new(Some(0)),
            }
        }

        pub fn full() -> Self {
            let bits = vec![u64::MAX; BITS_LEN]
                .into_boxed_slice()
                .try_into()
                .unwrap();

            Self {
                bits,
                length: Cell::new(Some(u16::MAX)),
            }
        }

        pub fn is_empty(&self) -> bool {
            let is_empty = arch::all_bits_are_unset(self.as_array());
            if is_empty {
                self.length.set(Some(0));
            }

            is_empty
        }

        pub fn is_full(&self) -> bool {
            let is_full = arch::all_bits_are_set(self.as_array());
            if is_full {
                self.length.set(Some(u16::MAX));
            }

            is_full
        }

        pub fn len(&self) -> usize {
            let length = self
                .length
                .get()
                .unwrap_or_else(|| arch::popcount(self.as_array()));
            self.length.set(Some(length));

            length as usize
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

        pub fn insert_all(&mut self) {
            self.length.set(Some(u16::MAX));
            arch::set_bits_one(self.as_mut_array());
        }

        pub fn remove_all(&mut self) {
            self.length.set(Some(0));
            arch::set_bits_zero(self.as_mut_array());
        }

        pub fn union(&mut self, other: &Self) {
            self.length.set(None);
            arch::union(self.as_mut_array(), other.as_array())
        }

        pub fn union_into(&self, other: &Self, output: &mut Self) {
            output.length.set(None);
            arch::union_into(self.as_array(), other.as_array(), output.as_mut_array())
        }

        pub fn union_new(&self, other: &Self) -> Self {
            let mut uninit = Box::new_uninit();
            arch::union_into_uninit(self.as_array(), other.as_array(), unsafe {
                transmute::<_, &mut [MaybeUninit<u64>; BITS_LEN]>(&mut *uninit)
            });

            Self {
                bits: unsafe { uninit.assume_init() },
                length: Cell::new(None),
            }
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
            let mut uninit = Box::new_uninit();
            arch::intersect_into_uninit(self.as_array(), other.as_array(), unsafe {
                transmute::<_, &mut [MaybeUninit<u64>; BITS_LEN]>(&mut *uninit)
            });

            Self {
                bits: unsafe { uninit.assume_init() },
                length: Cell::new(None),
            }
        }

        pub fn intersects(&self, other: &Self) -> bool {
            arch::intersects(self.as_array(), other.as_array())
        }

        pub fn is_disjoint(&self, other: &Self) -> bool {
            arch::is_disjoint(self.as_array(), other.as_array())
        }

        pub fn is_subset(&self, other: &Self) -> bool {
            arch::is_subset(self.as_array(), other.as_array())
        }

        fn as_array(&self) -> &[u64; BITS_LEN] {
            &*self.bits
        }

        fn as_mut_array(&mut self) -> &mut [u64; BITS_LEN] {
            &mut *self.bits
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

    mod arch {
        use super::BITS_LEN;
        use std::{
            mem::{size_of, transmute, MaybeUninit},
            simd::{
                mask64x4, simd_swizzle, u64x4,
                Which::{First, Second},
            },
            sync::atomic::{AtomicPtr, Ordering},
        };

        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        pub(super) fn set_bits_one(bits: &mut [u64; BITS_LEN]) {
            bits.fill(u64::MAX);
        }

        pub(super) fn set_bits_zero(bits: &mut [u64; BITS_LEN]) {
            bits.fill(0);
        }

        pub(super) fn all_bits_are_unset(bits: &[u64; BITS_LEN]) -> bool {
            let chunks = bits.array_chunks::<4>();
            debug_assert_eq!(chunks.remainder().len(), 0);

            let (mut all_are_unset, unset) = (mask64x4::splat(true), u64x4::splat(0));
            for (idx, &chunk) in chunks.into_iter().enumerate() {
                let mut chunk = u64x4::from_array(chunk);

                // Skip the very last u64
                if idx == 1023 {
                    chunk = simd_swizzle!(chunk, unset, [First(0), First(1), First(2), Second(3)]);
                }

                all_are_unset &= chunk.lanes_eq(unset);
            }

            all_are_unset.all()
        }

        pub(super) fn all_bits_are_set(bits: &[u64; BITS_LEN]) -> bool {
            let chunks = bits.array_chunks::<4>();
            debug_assert_eq!(chunks.remainder().len(), 0);

            let (set, mut all_are_set) = (u64x4::splat(u64::MAX), mask64x4::splat(true));

            for (idx, &chunk) in chunks.into_iter().enumerate() {
                let mut chunk = u64x4::from_array(chunk);

                // Skip the very last u64
                if idx == 1023 {
                    chunk = simd_swizzle!(chunk, set, [First(0), First(1), First(2), Second(3)]);
                }

                all_are_set &= chunk.lanes_eq(set);
            }

            all_are_set.all()
        }

        pub(crate) fn union(lhs: &mut [u64; BITS_LEN], rhs: &[u64; BITS_LEN]) {
            let (lhs, rhs) = (lhs.array_chunks_mut::<4>(), rhs.array_chunks::<4>());
            debug_assert_eq!(rhs.remainder().len(), 0);

            lhs.zip(rhs).for_each(|(lhs, &rhs)| {
                let lhs = unsafe { transmute::<&mut [u64; 4], &mut u64x4>(lhs) };
                let rhs = u64x4::from_array(rhs);
                *lhs |= rhs;
            });
        }

        pub(crate) fn union_into(
            lhs: &[u64; BITS_LEN],
            rhs: &[u64; BITS_LEN],
            output: &mut [u64; BITS_LEN],
        ) {
            let (lhs, rhs, output) = (
                lhs.array_chunks::<4>(),
                rhs.array_chunks::<4>(),
                output.array_chunks_mut::<4>(),
            );
            debug_assert_eq!(lhs.remainder().len(), 0);

            lhs.zip(rhs).zip(output).for_each(|((&lhs, &rhs), output)| {
                let (lhs, rhs) = (u64x4::from_array(lhs), u64x4::from_array(rhs));
                *output = unsafe { transmute::<u64x4, [u64; 4]>(lhs | rhs) };
            });
        }

        pub(crate) fn union_into_uninit(
            lhs: &[u64; BITS_LEN],
            rhs: &[u64; BITS_LEN],
            output: &mut [MaybeUninit<u64>; BITS_LEN],
        ) {
            let (lhs, rhs, output) = (
                lhs.array_chunks::<4>(),
                rhs.array_chunks::<4>(),
                output.array_chunks_mut::<4>(),
            );
            debug_assert_eq!(lhs.remainder().len(), 0);

            lhs.zip(rhs).zip(output).for_each(|((&lhs, &rhs), output)| {
                let (lhs, rhs) = (u64x4::from_array(lhs), u64x4::from_array(rhs));
                MaybeUninit::write_slice(output, (lhs | rhs).as_array());
            });
        }

        pub(crate) fn intersect(lhs: &mut [u64; BITS_LEN], rhs: &[u64; BITS_LEN]) {
            let (lhs, rhs) = (lhs.array_chunks_mut::<4>(), rhs.array_chunks::<4>());
            debug_assert_eq!(rhs.remainder().len(), 0);

            lhs.zip(rhs).for_each(|(lhs, &rhs)| {
                let lhs = unsafe { transmute::<&mut [u64; 4], &mut u64x4>(lhs) };
                let rhs = u64x4::from_array(rhs);
                *lhs &= rhs;
            });
        }

        pub(crate) fn intersect_into(
            lhs: &[u64; BITS_LEN],
            rhs: &[u64; BITS_LEN],
            output: &mut [u64; BITS_LEN],
        ) {
            let (lhs, rhs, output) = (
                lhs.array_chunks::<4>(),
                rhs.array_chunks::<4>(),
                output.array_chunks_mut::<4>(),
            );
            debug_assert_eq!(lhs.remainder().len(), 0);

            lhs.zip(rhs).zip(output).for_each(|((&lhs, &rhs), output)| {
                let (lhs, rhs) = (u64x4::from_array(lhs), u64x4::from_array(rhs));
                *output = *(lhs & rhs).as_array();
            });
        }

        pub(crate) fn intersect_into_uninit(
            lhs: &[u64; BITS_LEN],
            rhs: &[u64; BITS_LEN],
            output: &mut [MaybeUninit<u64>; BITS_LEN],
        ) {
            let (lhs, rhs, output) = (
                lhs.array_chunks::<4>(),
                rhs.array_chunks::<4>(),
                output.array_chunks_mut::<4>(),
            );
            debug_assert_eq!(lhs.remainder().len(), 0);

            lhs.zip(rhs).zip(output).for_each(|((&lhs, &rhs), output)| {
                let (lhs, rhs) = (u64x4::from_array(lhs), u64x4::from_array(rhs));
                MaybeUninit::write_slice(output, (lhs & rhs).as_array());
            });
        }

        pub(crate) fn is_disjoint(lhs: &[u64; BITS_LEN], rhs: &[u64; BITS_LEN]) -> bool {
            let (lhs, rhs) = (lhs.array_chunks::<4>(), rhs.array_chunks::<4>());
            debug_assert_eq!(lhs.remainder().len(), 0);

            let (mut is_disjoint, zero) = (mask64x4::splat(true), u64x4::splat(0));
            lhs.zip(rhs).for_each(|(&lhs, &rhs)| {
                let (lhs, rhs) = (u64x4::from_array(lhs), u64x4::from_array(rhs));
                is_disjoint &= (lhs & rhs).lanes_eq(zero);
            });

            is_disjoint.all()
        }

        pub(crate) fn is_subset(lhs: &[u64; BITS_LEN], rhs: &[u64; BITS_LEN]) -> bool {
            let (lhs, rhs) = (lhs.array_chunks::<4>(), rhs.array_chunks::<4>());
            debug_assert_eq!(lhs.remainder().len(), 0);

            let mut is_subset = mask64x4::splat(true);
            lhs.zip(rhs).for_each(|(&lhs, &rhs)| {
                let (lhs, rhs) = (u64x4::from_array(lhs), u64x4::from_array(rhs));
                is_subset &= (lhs & rhs).lanes_eq(lhs);
            });

            is_subset.all()
        }

        pub(crate) fn intersects(lhs: &[u64; BITS_LEN], rhs: &[u64; BITS_LEN]) -> bool {
            let (lhs, rhs) = (lhs.array_chunks::<4>(), rhs.array_chunks::<4>());
            debug_assert_eq!(lhs.remainder().len(), 0);

            let (mut intersects, zero) = (mask64x4::splat(false), u64x4::splat(0));
            lhs.zip(rhs).for_each(|(&lhs, &rhs)| {
                let (lhs, rhs) = (u64x4::from_array(lhs), u64x4::from_array(rhs));
                intersects |= (lhs & rhs).lanes_ne(zero);
            });

            intersects.any()
        }

        type PopcountPtr = unsafe fn(&[u64; BITS_LEN]) -> u16;

        pub(crate) fn popcount(bits: &[u64; BITS_LEN]) -> u16 {
            static POPCOUNT: AtomicPtr<()> = AtomicPtr::new(select_popcount as *mut ());

            unsafe fn select_popcount(bits: &[u64; BITS_LEN]) -> u16 {
                let selected: PopcountPtr = if is_x86_feature_detected!("avx2") {
                    popcount_avx2 as _
                } else {
                    popcount_scalar as _
                };
                POPCOUNT.store(selected as *mut (), Ordering::Relaxed);

                unsafe { selected(bits) }
            }

            let popcount = POPCOUNT.load(Ordering::Relaxed);
            let popcount = unsafe { transmute::<*mut (), PopcountPtr>(popcount) };
            unsafe { popcount(bits) }
        }

        pub(crate) fn popcount_scalar(bits: &[u64; BITS_LEN]) -> u16 {
            (bits.iter().map(|&bits| bits.count_ones()).sum::<u32>()
                - unsafe { bits.last().unwrap_unchecked().count_ones() }) as u16
        }

        // Based off of https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-avx2-lookup.cpp#L1-L61
        #[target_feature(enable = "avx2")]
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        pub(crate) unsafe fn popcount_avx2(bits: &[u64; BITS_LEN]) -> u16 {
            unsafe {
                // Get the pointer to the data and its length in bytes
                // We subtract 1 from BITS_LEN since we want to skip the very last
                // u64 in `bits` since it only exists for padding
                let (data_ptr, length) = (
                    bits.as_ptr().cast::<u8>(),
                    (BITS_LEN - 1) * size_of::<u64>(),
                );

                #[rustfmt::skip]
                let lookup = _mm256_setr_epi8(
                    /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
                    /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
                    /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
                    /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,

                    /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
                    /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
                    /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
                    /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,
                );

                let low_mask = _mm256_set1_epi8(0x0F);
                let zero = _mm256_setzero_si256();

                macro_rules! iter {
                    ($local:ident, $lookup:ident, $low_mask:ident, $data_ptr:ident, $idx:ident) => {{
                        let vec = _mm256_loadu_si256($data_ptr.add($idx).cast());

                        let lo = _mm256_and_si256(vec, $low_mask);
                        let hi = _mm256_and_si256(_mm256_srli_epi16(vec, 4), $low_mask);

                        let popcnt1 = _mm256_shuffle_epi8($lookup, lo);
                        let popcnt2 = _mm256_shuffle_epi8($lookup, hi);

                        $local = _mm256_add_epi8($local, popcnt1);
                        $local = _mm256_add_epi8($local, popcnt2);

                        $idx += 32;
                    }};
                }

                // Process the data in chunks of 8
                let (mut idx, mut acc) = (0, _mm256_setzero_si256());
                while idx + (8 * 32) <= length {
                    let mut local = zero;

                    iter!(local, lookup, low_mask, data_ptr, idx);
                    iter!(local, lookup, low_mask, data_ptr, idx);
                    iter!(local, lookup, low_mask, data_ptr, idx);
                    iter!(local, lookup, low_mask, data_ptr, idx);
                    iter!(local, lookup, low_mask, data_ptr, idx);
                    iter!(local, lookup, low_mask, data_ptr, idx);
                    iter!(local, lookup, low_mask, data_ptr, idx);
                    iter!(local, lookup, low_mask, data_ptr, idx);

                    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, zero));
                }

                // Finish off any remainder (will be <= 7 iterations)
                let mut local = zero;
                while idx + 32 <= length {
                    iter!(local, lookup, low_mask, data_ptr, idx);
                }
                acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, zero));

                // Sum up the aggregated popcount
                let mut result = _mm256_extract_epi64(acc, 0) as u64;
                result += _mm256_extract_epi64(acc, 1) as u64;
                result += _mm256_extract_epi64(acc, 2) as u64;
                result += _mm256_extract_epi64(acc, 3) as u64;

                // Finish off any uneven bits
                while idx < length {
                    result += POPCOUNT_LOOKUP[*data_ptr.add(idx) as usize] as u64;
                    idx += 1;
                }

                debug_assert!(u16::try_from(result).is_ok());
                result as u16
            }
        }

        #[rustfmt::skip]
        const POPCOUNT_LOOKUP: [u8; 256] = [
            /* 0 */ 0,  /* 1 */ 1,  /* 2 */ 1,  /* 3 */ 2,
            /* 4 */ 1,  /* 5 */ 2,  /* 6 */ 2,  /* 7 */ 3,
            /* 8 */ 1,  /* 9 */ 2,  /* a */ 2,  /* b */ 3,
            /* c */ 2,  /* d */ 3,  /* e */ 3,  /* f */ 4,
            /* 10 */ 1, /* 11 */ 2, /* 12 */ 2, /* 13 */ 3,
            /* 14 */ 2, /* 15 */ 3, /* 16 */ 3, /* 17 */ 4,
            /* 18 */ 2, /* 19 */ 3, /* 1a */ 3, /* 1b */ 4,
            /* 1c */ 3, /* 1d */ 4, /* 1e */ 4, /* 1f */ 5,
            /* 20 */ 1, /* 21 */ 2, /* 22 */ 2, /* 23 */ 3,
            /* 24 */ 2, /* 25 */ 3, /* 26 */ 3, /* 27 */ 4,
            /* 28 */ 2, /* 29 */ 3, /* 2a */ 3, /* 2b */ 4,
            /* 2c */ 3, /* 2d */ 4, /* 2e */ 4, /* 2f */ 5,
            /* 30 */ 2, /* 31 */ 3, /* 32 */ 3, /* 33 */ 4,
            /* 34 */ 3, /* 35 */ 4, /* 36 */ 4, /* 37 */ 5,
            /* 38 */ 3, /* 39 */ 4, /* 3a */ 4, /* 3b */ 5,
            /* 3c */ 4, /* 3d */ 5, /* 3e */ 5, /* 3f */ 6,
            /* 40 */ 1, /* 41 */ 2, /* 42 */ 2, /* 43 */ 3,
            /* 44 */ 2, /* 45 */ 3, /* 46 */ 3, /* 47 */ 4,
            /* 48 */ 2, /* 49 */ 3, /* 4a */ 3, /* 4b */ 4,
            /* 4c */ 3, /* 4d */ 4, /* 4e */ 4, /* 4f */ 5,
            /* 50 */ 2, /* 51 */ 3, /* 52 */ 3, /* 53 */ 4,
            /* 54 */ 3, /* 55 */ 4, /* 56 */ 4, /* 57 */ 5,
            /* 58 */ 3, /* 59 */ 4, /* 5a */ 4, /* 5b */ 5,
            /* 5c */ 4, /* 5d */ 5, /* 5e */ 5, /* 5f */ 6,
            /* 60 */ 2, /* 61 */ 3, /* 62 */ 3, /* 63 */ 4,
            /* 64 */ 3, /* 65 */ 4, /* 66 */ 4, /* 67 */ 5,
            /* 68 */ 3, /* 69 */ 4, /* 6a */ 4, /* 6b */ 5,
            /* 6c */ 4, /* 6d */ 5, /* 6e */ 5, /* 6f */ 6,
            /* 70 */ 3, /* 71 */ 4, /* 72 */ 4, /* 73 */ 5,
            /* 74 */ 4, /* 75 */ 5, /* 76 */ 5, /* 77 */ 6,
            /* 78 */ 4, /* 79 */ 5, /* 7a */ 5, /* 7b */ 6,
            /* 7c */ 5, /* 7d */ 6, /* 7e */ 6, /* 7f */ 7,
            /* 80 */ 1, /* 81 */ 2, /* 82 */ 2, /* 83 */ 3,
            /* 84 */ 2, /* 85 */ 3, /* 86 */ 3, /* 87 */ 4,
            /* 88 */ 2, /* 89 */ 3, /* 8a */ 3, /* 8b */ 4,
            /* 8c */ 3, /* 8d */ 4, /* 8e */ 4, /* 8f */ 5,
            /* 90 */ 2, /* 91 */ 3, /* 92 */ 3, /* 93 */ 4,
            /* 94 */ 3, /* 95 */ 4, /* 96 */ 4, /* 97 */ 5,
            /* 98 */ 3, /* 99 */ 4, /* 9a */ 4, /* 9b */ 5,
            /* 9c */ 4, /* 9d */ 5, /* 9e */ 5, /* 9f */ 6,
            /* a0 */ 2, /* a1 */ 3, /* a2 */ 3, /* a3 */ 4,
            /* a4 */ 3, /* a5 */ 4, /* a6 */ 4, /* a7 */ 5,
            /* a8 */ 3, /* a9 */ 4, /* aa */ 4, /* ab */ 5,
            /* ac */ 4, /* ad */ 5, /* ae */ 5, /* af */ 6,
            /* b0 */ 3, /* b1 */ 4, /* b2 */ 4, /* b3 */ 5,
            /* b4 */ 4, /* b5 */ 5, /* b6 */ 5, /* b7 */ 6,
            /* b8 */ 4, /* b9 */ 5, /* ba */ 5, /* bb */ 6,
            /* bc */ 5, /* bd */ 6, /* be */ 6, /* bf */ 7,
            /* c0 */ 2, /* c1 */ 3, /* c2 */ 3, /* c3 */ 4,
            /* c4 */ 3, /* c5 */ 4, /* c6 */ 4, /* c7 */ 5,
            /* c8 */ 3, /* c9 */ 4, /* ca */ 4, /* cb */ 5,
            /* cc */ 4, /* cd */ 5, /* ce */ 5, /* cf */ 6,
            /* d0 */ 3, /* d1 */ 4, /* d2 */ 4, /* d3 */ 5,
            /* d4 */ 4, /* d5 */ 5, /* d6 */ 5, /* d7 */ 6,
            /* d8 */ 4, /* d9 */ 5, /* da */ 5, /* db */ 6,
            /* dc */ 5, /* dd */ 6, /* de */ 6, /* df */ 7,
            /* e0 */ 3, /* e1 */ 4, /* e2 */ 4, /* e3 */ 5,
            /* e4 */ 4, /* e5 */ 5, /* e6 */ 5, /* e7 */ 6,
            /* e8 */ 4, /* e9 */ 5, /* ea */ 5, /* eb */ 6,
            /* ec */ 5, /* ed */ 6, /* ee */ 6, /* ef */ 7,
            /* f0 */ 4, /* f1 */ 5, /* f2 */ 5, /* f3 */ 6,
            /* f4 */ 5, /* f5 */ 6, /* f6 */ 6, /* f7 */ 7,
            /* f8 */ 5, /* f9 */ 6, /* fa */ 6, /* fb */ 7,
            /* fc */ 6, /* fd */ 7, /* fe */ 7, /* ff */ 8,
        ];
    }
}
