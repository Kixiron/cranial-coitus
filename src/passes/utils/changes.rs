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
