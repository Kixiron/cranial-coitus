#[inline]
pub fn lsb(chunk: u64) -> Option<u8> {
    if chunk == 0 {
        None
    } else {
        Some(chunk.trailing_zeros() as u8)
    }
}

/// Returns the last (most significant) bit of `chunk`, or `None` if `chunk` is
/// 0.
#[inline]
pub fn msb(chunk: u64) -> Option<u8> {
    if chunk == 0 {
        None
    } else {
        let bits = u64::BITS - 1;
        Some((bits as u8) ^ chunk.leading_zeros() as u8)
    }
}

/// Removes the first (least significant) bit from `chunk` and returns it, or
/// `None` if `chunk` is 0.
#[inline]
pub fn pop_lsb(chunk: &mut u64) -> Option<u8> {
    let lsb = lsb(*chunk)?;
    *chunk ^= 1 << lsb;
    Some(lsb)
}

/// Removes the last (most significant) bit from `chunk` and returns it, or
/// `None` if `chunk` is 0.
#[inline]
pub fn pop_msb(chunk: &mut u64) -> Option<u8> {
    let msb = msb(*chunk)?;
    *chunk ^= 1 << msb;
    Some(msb)
}
