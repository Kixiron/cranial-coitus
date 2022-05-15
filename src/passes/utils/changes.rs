use crate::utils::{AssertNone, HashMap};

pub type ChangeReport = HashMap<&'static str, usize>;

#[derive(Debug)]
pub struct Changes<const LEN: usize> {
    changes: [(&'static str, usize); LEN],
    has_changed: bool,
}

impl<const LEN: usize> Changes<LEN> {
    #[must_use]
    pub fn new(changes: [&'static str; LEN]) -> Self {
        Self {
            changes: changes.map(|name| (name, 0)),
            has_changed: false,
        }
    }

    pub fn did_change(&self) -> bool {
        self.has_changed
    }

    pub fn reset(&mut self) {
        self.has_changed = false;
    }

    pub fn combine(&mut self, other: &Self) {
        self.has_changed |= other.has_changed;

        debug_assert_eq!(
            self.changes.map(|(name, _)| name),
            other.changes.map(|(name, _)| name),
            "tried to combine unequal changes",
        );

        // Sum up all changes
        for ((_, total), &(_, additional)) in self.changes.iter_mut().zip(&other.changes) {
            *total += additional;
        }
    }

    pub fn inc<const NAME: &'static str>(&mut self) {
        self.has_changed = true;
        for (name, total) in &mut self.changes {
            if *name == NAME {
                *total += 1;
                break;
            }
        }
    }

    pub fn as_report(&self) -> ChangeReport {
        let mut report = HashMap::with_capacity_and_hasher(LEN, Default::default());
        for (name, total) in self.changes {
            report.insert(name, total).debug_unwrap_none();
        }

        report
    }
}

/* An attempt to make the names checked at compile time, currently ICEs

mod const_checked {
    use crate::{
        passes::utils::ChangeReport,
        utils::{AssertNone, HashMap},
    };

    pub struct Changes<const CHANGES: &'static [&'static str]>
    where
        [(); CHANGES.len()]:,
    {
        changes: [usize; CHANGES.len()],
        has_changed: bool,
    }

    impl<const CHANGES: &'static [&'static str]> Changes<CHANGES>
    where
        [(); CHANGES.len()]:,
    {
        pub const fn new() -> Self {
            Self {
                changes: [0; CHANGES.len()],
                has_changed: false,
            }
        }

        pub const fn did_change(&self) -> bool {
            self.has_changed
        }

        pub fn reset(&mut self) {
            self.has_changed = false;
        }

        pub fn combine(&mut self, other: &Self) {
            self.has_changed |= other.has_changed;

            // Sum up all changes
            self.changes
                .iter_mut()
                .zip(&other.changes)
                .for_each(|(total, &additional)| *total += additional);
        }

        pub fn inc<const NAME: &'static str>(&mut self)
        where
            Assert<{ contains(CHANGES, NAME) }>: True,
        {
            self.has_changed = true;

            let index = index_of(CHANGES, NAME);
            self.changes[index] += 1;
        }

        pub fn as_report(&self) -> ChangeReport {
            let mut report = HashMap::with_capacity_and_hasher(CHANGES.len(), Default::default());
            for (idx, &total) in self.changes.iter().enumerate() {
                report.insert(CHANGES[idx], total).debug_unwrap_none();
            }

            report
        }
    }

    pub struct Assert<const COND: bool> {}

    pub trait True {}

    impl True for Assert<true> {}

    pub const fn contains(changes: &[&str], name: &str) -> bool {
        let mut idx = 0;
        while idx < changes.len() {
            if str_eq(changes[idx], name) {
                return true;
            }

            idx += 1;
        }

        false
    }

    const fn index_of(changes: &[&str], name: &str) -> usize {
        let mut idx = 0;
        while idx < changes.len() {
            if str_eq(changes[idx], name) {
                return idx;
            }

            idx += 1;
        }

        panic!()
    }

    const fn str_eq(lhs: &str, rhs: &str) -> bool {
        let (lhs, rhs) = (lhs.as_bytes(), rhs.as_bytes());

        if lhs.len() != rhs.len() {
            return false;
        }

        let mut idx = 0;
        while idx < lhs.len() {
            if lhs[idx] != rhs[idx] {
                return false;
            }

            idx += 1;
        }

        true
    }
}
*/
