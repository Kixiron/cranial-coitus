use crate::passes::dataflow::domain::bitmap_u16::{BitArray, BITS_LEN};
use std::{
    mem::{size_of, transmute, MaybeUninit},
    simd::{mask64x4, u64x4},
    sync::atomic::{AtomicPtr, Ordering},
};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(super) fn set_bits_one(bits: &mut BitArray) {
    bits.fill(u64::MAX);
}

pub(super) fn set_uninit_bits_one(bits: &mut [MaybeUninit<u64>; BITS_LEN]) {
    bits.fill(MaybeUninit::new(u64::MAX));
}

pub(super) fn set_bits_zero(bits: &mut BitArray) {
    bits.fill(0);
}

pub(super) fn all_bits_are_unset(bits: &BitArray) -> bool {
    let chunks = bits.array_chunks::<4>();
    debug_assert_eq!(chunks.remainder().len(), 0);

    let (mut all_are_unset, unset) = (mask64x4::splat(true), u64x4::splat(0));
    chunks.for_each(|&chunk| all_are_unset &= u64x4::from_array(chunk).lanes_eq(unset));

    all_are_unset.all()
}

pub(super) fn all_bits_are_set(bits: &BitArray) -> bool {
    let chunks = bits.array_chunks::<4>();
    debug_assert_eq!(chunks.remainder().len(), 0);

    let (set, mut all_are_set) = (u64x4::splat(u64::MAX), mask64x4::splat(true));
    chunks.for_each(|&chunk| all_are_set &= u64x4::from_array(chunk).lanes_eq(set));

    all_are_set.all()
}

pub(crate) fn union(lhs: &mut BitArray, rhs: &BitArray) -> bool {
    let (lhs, rhs) = (lhs.array_chunks_mut::<4>(), rhs.array_chunks::<4>());
    debug_assert_eq!(rhs.remainder().len(), 0);

    let mut changed = mask64x4::splat(false);
    lhs.zip(rhs).for_each(|(lhs, &rhs)| {
        let lhs = unsafe { transmute::<&mut [u64; 4], &mut u64x4>(lhs) };
        let old = *lhs;
        let rhs = u64x4::from_array(rhs);

        *lhs |= rhs;
        changed |= lhs.lanes_ne(old);
    });

    changed.any()
}

pub(crate) fn union_into(lhs: &BitArray, rhs: &BitArray, output: &mut BitArray) -> bool {
    let (lhs, rhs, output) = (
        lhs.array_chunks::<4>(),
        rhs.array_chunks::<4>(),
        output.array_chunks_mut::<4>(),
    );
    debug_assert_eq!(lhs.remainder().len(), 0);

    let mut changed = mask64x4::splat(false);
    lhs.zip(rhs).zip(output).for_each(|((&lhs, &rhs), output)| {
        let (lhs, rhs) = (u64x4::from_array(lhs), u64x4::from_array(rhs));
        *output = *(lhs | rhs).as_array();
        changed |= lhs.lanes_ne(u64x4::from_array(*output));
    });

    changed.any()
}

pub(crate) fn union_into_uninit(
    lhs: &BitArray,
    rhs: &BitArray,
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

pub(crate) fn intersect(lhs: &mut BitArray, rhs: &BitArray) {
    let (lhs, rhs) = (lhs.array_chunks_mut::<4>(), rhs.array_chunks::<4>());
    debug_assert_eq!(rhs.remainder().len(), 0);

    lhs.zip(rhs).for_each(|(lhs, &rhs)| {
        let lhs = unsafe { transmute::<&mut [u64; 4], &mut u64x4>(lhs) };
        let rhs = u64x4::from_array(rhs);
        *lhs &= rhs;
    });
}

pub(crate) fn intersect_into(lhs: &BitArray, rhs: &BitArray, output: &mut BitArray) {
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
    lhs: &BitArray,
    rhs: &BitArray,
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

pub(crate) fn is_disjoint(lhs: &BitArray, rhs: &BitArray) -> bool {
    let (lhs, rhs) = (lhs.array_chunks::<4>(), rhs.array_chunks::<4>());
    debug_assert_eq!(lhs.remainder().len(), 0);

    let (mut is_disjoint, zero) = (mask64x4::splat(true), u64x4::splat(0));
    lhs.zip(rhs).for_each(|(&lhs, &rhs)| {
        let (lhs, rhs) = (u64x4::from_array(lhs), u64x4::from_array(rhs));
        is_disjoint &= (lhs & rhs).lanes_eq(zero);
    });

    is_disjoint.all()
}

pub(crate) fn is_subset(lhs: &BitArray, rhs: &BitArray) -> bool {
    let (lhs, rhs) = (lhs.array_chunks::<4>(), rhs.array_chunks::<4>());
    debug_assert_eq!(lhs.remainder().len(), 0);

    let mut is_subset = mask64x4::splat(true);
    lhs.zip(rhs).for_each(|(&lhs, &rhs)| {
        let (lhs, rhs) = (u64x4::from_array(lhs), u64x4::from_array(rhs));
        is_subset &= (lhs & rhs).lanes_eq(lhs);
    });

    is_subset.all()
}

pub(crate) fn intersects(lhs: &BitArray, rhs: &BitArray) -> bool {
    let (lhs, rhs) = (lhs.array_chunks::<4>(), rhs.array_chunks::<4>());
    debug_assert_eq!(lhs.remainder().len(), 0);

    let (mut intersects, zero) = (mask64x4::splat(false), u64x4::splat(0));
    lhs.zip(rhs).for_each(|(&lhs, &rhs)| {
        let (lhs, rhs) = (u64x4::from_array(lhs), u64x4::from_array(rhs));
        intersects |= (lhs & rhs).lanes_ne(zero);
    });

    intersects.any()
}

pub(super) fn bitmap_eq(lhs: &BitArray, rhs: &BitArray) -> bool {
    let (lhs, rhs) = (lhs.array_chunks::<4>(), rhs.array_chunks::<4>());
    debug_assert_eq!(lhs.remainder().len(), 0);
    debug_assert_eq!(rhs.remainder().len(), 0);

    let mut are_equal = mask64x4::splat(true);
    lhs.zip(rhs).for_each(|(&lhs, &rhs)| {
        let (lhs, rhs) = (u64x4::from_array(lhs), u64x4::from_array(rhs));
        are_equal &= lhs.lanes_eq(rhs);
    });

    are_equal.all()
}

type PopcountPtr = unsafe fn(&BitArray) -> u32;

pub(super) fn popcount(bits: &BitArray) -> u32 {
    static POPCOUNT: AtomicPtr<()> = AtomicPtr::new(select_popcount as *mut ());

    unsafe fn select_popcount(bits: &BitArray) -> u32 {
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

pub(crate) fn popcount_scalar(bits: &BitArray) -> u32 {
    bits.iter().map(|&bits| bits.count_ones()).sum::<u32>() as u32
}

// Based off of https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-avx2-lookup.cpp#L1-L61
#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) unsafe fn popcount_avx2(bits: &BitArray) -> u32 {
    unsafe {
        // Get the pointer to the data and its length in bytes
        let (data_ptr, length) = (bits.as_ptr().cast::<u8>(), BITS_LEN * size_of::<u64>());

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

        debug_assert!(u32::try_from(result).is_ok());
        result as u32
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

#[cfg(test)]
mod tests {
    use crate::passes::dataflow::domain::bitmap_u16::{
        arch::{popcount, popcount_avx2, popcount_scalar},
        U16Bitmap, MAX_LEN,
    };

    #[track_caller]
    fn run_popcount(bitmap: &U16Bitmap, expected_len: u32) {
        let len = popcount(bitmap.as_array());
        assert_eq!(len, expected_len);

        let len = popcount_scalar(bitmap.as_array());
        assert_eq!(len, expected_len);

        if is_x86_feature_detected!("avx2") {
            let len = unsafe { popcount_avx2(bitmap.as_array()) };
            assert_eq!(len, expected_len);
        }
    }

    #[test]
    fn bitmap_popcount() {
        let empty = U16Bitmap::empty();
        run_popcount(&empty, 0);

        let full = U16Bitmap::full();
        run_popcount(&full, MAX_LEN);

        let five_thousand = (0..10_000).step_by(2).collect();
        run_popcount(&five_thousand, 10_000 / 2);
    }
}
