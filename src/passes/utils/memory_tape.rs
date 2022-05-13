use crate::{
    graph::OutputPort,
    values::{Cell, Ptr},
};
use std::{
    cell::RefCell,
    mem::take,
    ops::{Index, IndexMut},
    thread,
};

thread_local! {
    // FIXME: https://github.com/rust-lang/rust-clippy/issues/8493
    #[allow(clippy::declare_interior_mutable_const)]
    static TAPE_BUFFERS: RefCell<Vec<Vec<MemoryCell>>>
        = const { RefCell::new(Vec::new()) };
}

#[derive(Debug)]
pub struct MemoryTape {
    tape: Vec<MemoryCell>,
}

impl MemoryTape {
    pub fn zeroed(tape_len: u16) -> Self {
        Self::new_with(tape_len, MemoryCell::Cell(Cell::zero()))
    }

    pub fn unknown(tape_len: u16) -> Self {
        Self::new_with(tape_len, MemoryCell::Unknown)
    }

    fn new_with(tape_len: u16, value: MemoryCell) -> Self {
        let tape = TAPE_BUFFERS
            .with_borrow_mut(|buffers| {
                let mut tape = buffers.pop();
                if let Some(tape) = tape.as_mut() {
                    debug_assert!(tape.is_empty());
                    tape.resize(tape_len as usize, value)
                }

                tape
            })
            .unwrap_or_else(|| vec![value; tape_len as usize]);

        Self { tape }
    }

    pub fn zero(&mut self) {
        self.tape.fill(MemoryCell::Cell(Cell::zero()));
    }

    pub fn mystify(&mut self) {
        self.tape.fill(MemoryCell::Unknown);
    }

    pub fn mapped<F>(&self, mut map: F) -> Self
    where
        F: FnMut(MemoryCell) -> MemoryCell,
    {
        let mut new = self.clone();
        new.tape.iter_mut().for_each(|value| *value = map(*value));

        new
    }

    pub fn tape_len(&self) -> u16 {
        self.tape.len() as u16
    }
}

impl Index<Ptr> for MemoryTape {
    type Output = MemoryCell;

    #[inline]
    fn index(&self, index: Ptr) -> &Self::Output {
        debug_assert_eq!(index.tape_len() as usize, self.tape.len());
        &self.tape[index.value() as usize]
    }
}

impl IndexMut<Ptr> for MemoryTape {
    #[inline]
    fn index_mut(&mut self, index: Ptr) -> &mut Self::Output {
        debug_assert_eq!(index.tape_len() as usize, self.tape.len());
        &mut self.tape[index.value() as usize]
    }
}

impl Clone for MemoryTape {
    fn clone(&self) -> Self {
        let tape = if let Some(mut tape) = TAPE_BUFFERS.with_borrow_mut(|buffers| buffers.pop()) {
            tape.clone_from(&self.tape);
            tape
        } else {
            self.tape.clone()
        };

        Self { tape }
    }

    fn clone_from(&mut self, source: &Self) {
        self.tape.clone_from(&source.tape);
    }
}

impl Drop for MemoryTape {
    fn drop(&mut self) {
        if !thread::panicking() {
            let mut tape = take(&mut self.tape);
            tape.clear();

            TAPE_BUFFERS.with_borrow_mut(|buffers| buffers.push(tape));
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryCell {
    Unknown,
    Cell(Cell),
    Port(OutputPort),
}

impl MemoryCell {
    pub const fn as_cell(&self) -> Option<Cell> {
        if let Self::Cell(cell) = *self {
            Some(cell)
        } else {
            None
        }
    }
}
