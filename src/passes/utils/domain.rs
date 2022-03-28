use core::fmt::{self, Debug};
use tinyvec::{ArrayVec, TinyVec};

struct IntegerDomain {
    values: TinyVec<[(u16, u16); 1]>,
}

impl IntegerDomain {
    pub const fn new() -> Self {
        Self {
            values: TinyVec::Inline(ArrayVec::from_array_empty([(0, 0); 1])),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: TinyVec::with_capacity(capacity),
        }
    }

    /// An alias for [`IntegerDomain::new()`]
    pub const fn empty() -> Self {
        Self::new()
    }

    pub fn full(tape_len: u16) -> Self {
        Self {
            values: TinyVec::Inline(ArrayVec::from_array_len([(0, tape_len - 1); 1], 1)),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn singleton(value: u16) -> Self {
        Self {
            values: TinyVec::Inline(ArrayVec::from_array_len([(value, value); 1], 1)),
        }
    }

    pub fn insert(&mut self, value: u16) {
        if self.values.is_empty() {
            self.values.push((value, value));
        } else {
            let mut idx = 0;
            while idx < self.values.len() {
                let (start, end) = unsafe { self.values.get_unchecked_mut(idx) };

                // If the value is contained within this range, we don't need to do anything else
                if *start <= value && *end >= value {
                    return;

                // If value is one below the range's start, extend the range
                } else if *start <= value + 1 && *end >= value {
                    *start = value;
                    return;

                // If value is one above the range's end, extend the range
                } else if *start <= value && *end + 1 >= value {
                    *end = value;
                    return;

                // If the start value is less than the value then we can insert a
                // new range right before it
                } else if *start > value {
                    self.values.insert(idx, (value, value));
                    return;

                // Otherwise continue searching
                } else {
                    idx += 1;
                }
            }

            // If we reach this, we searched all ranges and found no matches, so push a new range
            self.values.push((value, value));
        }
    }
}

impl Debug for IntegerDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct DebugRange {
            start: u16,
            end: u16,
        }

        impl Debug for DebugRange {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.start == self.end {
                    self.start.fmt(f)
                } else {
                    write!(f, "{}..{}", self.start, self.end)
                }
            }
        }

        f.debug_set()
            .entries(
                self.values
                    .iter()
                    .map(|&(start, end)| DebugRange { start, end }),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::IntegerDomain;

    #[test]
    fn insert_domain() {
        let mut domain = IntegerDomain::new();
        domain.insert(0);
        domain.insert(1);
        domain.insert(100);

        dbg!(domain);
    }
}
