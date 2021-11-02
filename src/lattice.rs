use std::cmp::{max, min};

/// A [partially ordered set] that has a [least upper bound] for
/// any pair of elements in the set.
///
/// [partially ordered set]: https://en.wikipedia.org/wiki/Partially_ordered_set
/// [least upper bound]: https://en.wikipedia.org/wiki/Infimum_and_supremum
pub trait SemiLattice: Eq {
    /// Computes the least upper bound of two elements, storing the result in `self` and returning
    /// `true` if `self` has changed.
    ///
    /// The lattice join operator is abbreviated as `∨`.
    fn join(&mut self, other: &Self);
}

/// A `bool` is a "two-point" lattice with `true` as the top element and `false` as the bottom:
///
/// ```text
///      true
///        |
///      false
/// ```
impl SemiLattice for bool {
    fn join(&mut self, other: &Self) {
        if matches!((*self, *other), (false, true)) {
            *self = true;
        }
    }
}

/// A tuple (or list) of lattices is itself a lattice whose bounds are the concatenation
/// of the bounds of each element of the tuple (or list).
///
/// In other words:
///     (A₀, A₁, ..., Aₙ) ∨ (B₀, B₁, ..., Bₙ) = (A₀ ∨ B₀, A₁ ∨ B₁, ..., Aₙ ∨ Bₙ)
impl<A> SemiLattice for (A,)
where
    A: SemiLattice,
{
    fn join(&mut self, other: &Self) {
        self.0.join(&other.0);
    }
}

/// A tuple (or list) of lattices is itself a lattice whose bounds are the concatenation
/// of the bounds of each element of the tuple (or list).
///
/// In other words:
///     (A₀, A₁, ..., Aₙ) ∨ (B₀, B₁, ..., Bₙ) = (A₀ ∨ B₀, A₁ ∨ B₁, ..., Aₙ ∨ Bₙ)
impl<A, B> SemiLattice for (A, B)
where
    A: SemiLattice,
    B: SemiLattice,
{
    fn join(&mut self, other: &Self) {
        self.0.join(&other.0);
        self.1.join(&other.1);
    }
}

/// A tuple (or list) of lattices is itself a lattice whose bounds are the concatenation
/// of the bounds of each element of the tuple (or list).
///
/// In other words:
///     (A₀, A₁, ..., Aₙ) ∨ (B₀, B₁, ..., Bₙ) = (A₀ ∨ B₀, A₁ ∨ B₁, ..., Aₙ ∨ Bₙ)
impl<A, B, C> SemiLattice for (A, B, C)
where
    A: SemiLattice,
    B: SemiLattice,
    C: SemiLattice,
{
    fn join(&mut self, other: &Self) {
        self.0.join(&other.0);
        self.1.join(&other.1);
        self.2.join(&other.2);
    }
}

impl<T> SemiLattice for Option<T>
where
    T: SemiLattice,
{
    fn join(&mut self, other: &Self) {
        match (self, other) {
            (None, _) => {}
            (this @ Some(_), None) => *this = None,
            (Some(lhs), Some(rhs)) => lhs.join(rhs),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RangeLattice<T> {
    Range { low: T, high: T },
    Bottom,
}

impl<T> RangeLattice<T> {
    pub const fn new(low: T, high: T) -> Self {
        Self::Range { low, high }
    }
}

impl<T> SemiLattice for RangeLattice<T>
where
    T: Ord + Clone,
{
    fn join(&mut self, other: &Self) {
        match (self, other) {
            (
                Self::Range {
                    low: low1,
                    high: high1,
                },
                Self::Range {
                    low: low2,
                    high: high2,
                },
            ) => {
                *low1 = max(low1.clone(), low2.clone());
                *high1 = min(high1.clone(), high2.clone());
            }
            (this, _) => *this = Self::Bottom,
        }
    }
}
