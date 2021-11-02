use std::mem;

#[derive(Debug, Clone, Default)]
pub struct UnionFind {
    parents: Vec<Unioned>,
}

impl UnionFind {
    pub fn new() -> Self {
        Self {
            parents: Vec::new(),
        }
    }

    pub fn make_set(&mut self) -> Unioned {
        // Safety: `id` is in-bounds of `parents`
        let id = unsafe { Unioned::new(self.parents.len()) };
        self.parents.push(id);

        id
    }

    pub fn reserve(&mut self, additional: usize) {
        self.parents.reserve(additional);
    }

    #[inline(always)]
    fn parent(&self, query: Unioned) -> Unioned {
        debug_assert!(query.index < self.parents.len());

        // Safety: `index` is in-bounds of `parents`
        unsafe { *self.parents.get_unchecked(query.index) }
    }

    #[inline(always)]
    fn set_parent(&mut self, query: Unioned, new_parent: Unioned) {
        debug_assert!(query.index < self.parents.len());

        // Safety: `index` is in-bounds of `parents`
        unsafe { *self.parents.get_unchecked_mut(query.index) = new_parent };
    }

    pub fn find(&mut self, mut current: Unioned) -> Unioned {
        loop {
            let parent = self.parent(current);
            if current == parent {
                return parent;
            }

            // do path halving and proceed
            let grandparent = self.parent(parent);
            self.set_parent(current, grandparent);

            current = grandparent;
        }
    }

    /// Returns (new_leader, old_leader)
    pub fn union(&mut self, set1: Unioned, set2: Unioned) -> (Unioned, Unioned) {
        let mut root1 = self.find(set1);
        let mut root2 = self.find(set2);

        if root1 == root2 {
            (root1, root2)
        } else {
            if root1 > root2 {
                mem::swap(&mut root1, &mut root2);
            }

            self.set_parent(root2, root1);

            (root1, root2)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Unioned {
    index: usize,
}

impl Unioned {
    /// # Safety
    ///
    /// `index` must be in-bounds of the parent [`UnionFind`]
    /// and must only be used with the `UnionFind` that created it
    pub const unsafe fn new(index: usize) -> Self {
        Self { index }
    }
}
