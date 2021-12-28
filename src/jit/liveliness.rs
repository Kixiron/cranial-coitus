use crate::jit::{
    basic_block::{BasicBlock, BlockId, Blocks, Instruction, Terminator, ValId, Value},
    block_visitor::BasicBlockVisitor,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{self, Debug},
};

type Location = (BlockId, Option<usize>);

pub struct Liveliness {
    first_usages: BTreeMap<ValId, Location>,
    last_usages: BTreeMap<ValId, Location>,
    current_block: BlockId,
    current_inst: Option<usize>,
}

impl Liveliness {
    pub fn new() -> Self {
        Self {
            first_usages: BTreeMap::new(),
            last_usages: BTreeMap::new(),
            current_block: BlockId::new(u32::MAX),
            current_inst: None,
        }
    }

    pub fn run(&mut self, blocks: &Blocks) {
        self.visit_blocks(blocks);
    }

    /// Returns `true` if the given instruction of the given block is the *first* usage of the given [`ValId`].
    /// A [`None`] given for `index` means a block terminator instead of an instruction in the block's body
    pub fn is_first_usage(&self, val: ValId, block: BlockId, index: Option<usize>) -> bool {
        self.first_usages.get(&val).copied() == Some((block, index))
    }

    /// Returns `true` if the given instruction of the given block is the *last* usage of the given [`ValId`].
    /// A [`None`] given for `index` means a block terminator instead of an instruction in the block's body
    pub fn is_last_usage(&self, val: ValId, block: BlockId, index: Option<usize>) -> bool {
        self.last_usages.get(&val).copied() == Some((block, index))
    }

    /// Returns `true` if the given instruction of the given block is the *only* usage of the given [`ValId`].
    /// A [`None`] given for `index` means a block terminator instead of an instruction in the block's body
    pub fn is_only_usage(&self, val: ValId, block: BlockId, index: Option<usize>) -> bool {
        // If an instruction is both the first and last usage of a value then it should be the only user.
        // This doesn't account for instructions that use a value twice, like `add %x, %x`, but that shouldn't
        // be a problem
        self.is_first_usage(val, block, index) && self.is_last_usage(val, block, index)
    }
}

impl BasicBlockVisitor for Liveliness {
    fn visit_value(&mut self, value: Value) {
        if let Value::Val(val) = value {
            // Only insert for first usages if this is indeed the first usage
            self.first_usages
                .entry(val)
                .or_insert((self.current_block, self.current_inst));

            // Unconditionally insert for last usages so that the final usage
            // is the one propagated
            self.last_usages
                .insert(val, (self.current_block, self.current_inst));
        }
    }

    fn before_visit_block(&mut self, block: &BasicBlock) {
        self.current_block = block.id();
    }

    fn before_visit_inst(&mut self, _inst: &Instruction) {
        match &mut self.current_inst {
            Some(idx) => *idx += 1,
            current @ None => *current = Some(0),
        }
    }

    fn before_visit_term(&mut self, _term: &Terminator) {
        self.current_inst = None;
    }
}

impl Debug for Liveliness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Liveliness")
            .field("first_usages", &DebugLocationMap(&self.first_usages))
            .field("last_usages", &DebugLocationMap(&self.last_usages))
            .finish_non_exhaustive()
    }
}

struct DebugLocationMap<'a>(&'a BTreeMap<ValId, Location>);

impl Debug for DebugLocationMap<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(
                self.0
                    .iter()
                    .map(|(&val, &loc)| (val.display(), DebugLocation(loc))),
            )
            .finish()
    }
}

struct DebugLocation(Location);

impl Debug for DebugLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (block, inst) = self.0;

        match inst {
            Some(idx) => write!(f, "({}, {})", block, idx),
            None => write!(f, "({}, term)", block),
        }
    }
}
