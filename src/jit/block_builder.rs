use crate::{
    ir::{Const, Value as CirValue, VarId},
    jit::basic_block::{BasicBlock, BlockId, Instruction, Terminator, ValId, Value},
    utils::AssertNone,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
};

#[derive(Debug, Clone)]
pub struct BlockBuilder {
    entry: BlockId,
    current: Option<BasicBlock>,
    pub(super) blocks: BTreeMap<BlockId, BasicBlock>,
    allocated_blocks: BTreeSet<BlockId>,
    values: BTreeMap<VarId, Value>,
    block_counter: u32,
    val_counter: u32,
}

/// Public API
impl BlockBuilder {
    /// Creates a new basic block builder
    pub fn new() -> Self {
        let entry = BlockId::new(0);

        Self {
            entry,
            current: Some(BasicBlock::new(entry, Vec::new(), Terminator::Error)),
            blocks: BTreeMap::new(),
            allocated_blocks: BTreeSet::new(),
            values: BTreeMap::new(),
            block_counter: 1,
            val_counter: 0,
        }
    }

    pub fn finalize(&mut self) {
        if let Some(current) = self.current.take() {
            self.blocks
                .insert(current.id(), current)
                .debug_unwrap_none();
        }
    }

    pub fn into_blocks(mut self) -> Vec<BasicBlock> {
        self.finalize();

        debug_assert!(self.allocated_blocks.is_empty());
        debug_assert!(self.blocks.contains_key(&self.entry));

        self.blocks.into_values().collect()
    }

    /// Allocates a basic block to be created in in the future
    pub fn allocate(&mut self) -> AllocatedBlock {
        let block = self.next_block();
        self.allocated_blocks.insert(block);

        AllocatedBlock::new(block)
    }

    /// Allocates multiple blocks to be created in the future
    pub fn allocate_blocks<const N: usize>(&mut self) -> [AllocatedBlock; N] {
        [(); N].map(|()| self.allocate())
    }

    /// Creates a block for the allocated block and sets the current block to it
    pub fn create_block(&mut self, block: AllocatedBlock) -> &mut BasicBlock {
        let was_allocated = self.allocated_blocks.remove(&block.block);
        debug_assert!(was_allocated);

        let block = BasicBlock::new(block.block, Vec::new(), Terminator::Error);
        self.replace_current(block);

        self.current.as_mut().unwrap()
    }

    /// Move to the given block
    #[allow(dead_code)]
    pub fn move_to(&mut self, block: BlockId) -> &mut BasicBlock {
        if self.current.as_ref().unwrap().id() != block {
            if let Some(block) = self.blocks.remove(&block) {
                self.replace_current(block);
            } else {
                panic!("tried to move to {} but it doesn't exist", block);
            }
        }

        self.current.as_mut().unwrap()
    }

    /// Pushes an instruction to the current block
    pub fn push<I>(&mut self, inst: I)
    where
        I: Into<Instruction>,
    {
        self.current().push(inst.into());
    }

    /// Assigns a SSA value to a variable id
    pub fn assign<V>(&mut self, var: VarId, value: V)
    where
        V: Into<Value>,
    {
        self.values.insert(var, value.into()).debug_unwrap_none();
    }

    /// Gets the current block
    pub fn current(&mut self) -> &mut BasicBlock {
        self.current.as_mut().unwrap()
    }

    /// Gets the id of the current block
    pub fn current_id(&mut self) -> BlockId {
        self.current().id()
    }

    /// Gets the SSA value associated with an input value
    #[track_caller]
    pub fn get(&self, value: CirValue) -> Value {
        match value {
            CirValue::Var(var) => *self.values.get(&var).unwrap(),
            CirValue::Const(constant) => match constant {
                Const::Int(uint) => Value::Uint(uint),
                Const::U8(byte) => Value::Byte(byte),
                Const::Bool(bool) => Value::Bool(bool),
            },
            CirValue::Missing => unreachable!(),
        }
    }

    /// Generates a unique value id
    pub fn create_val(&mut self) -> ValId {
        let val = self.val_counter;
        self.val_counter += 1;

        ValId::new(val)
    }

    pub const fn entry(&self) -> BlockId {
        self.entry
    }

    pub fn set_entry(&mut self, entry: BlockId) {
        self.entry = entry;
    }
}

/// Internal utility functions
impl BlockBuilder {
    /// Creates a new block with a unique id
    #[allow(dead_code)]
    fn new_block(&mut self) -> BasicBlock {
        BasicBlock::new(self.next_block(), Vec::new(), Terminator::Error)
    }

    /// Replaces the current block with the given one, putting the previous
    /// block into `self.blocks`
    fn replace_current(&mut self, block: BasicBlock) {
        let previous = mem::replace(self.current.as_mut().unwrap(), block);
        self.blocks
            .insert(previous.id(), previous)
            .debug_unwrap_none();
    }

    /// Generates a unique block id
    fn next_block(&mut self) -> BlockId {
        let block = self.block_counter;
        self.block_counter += 1;

        BlockId::new(block)
    }
}

#[derive(Debug)]
pub struct AllocatedBlock {
    block: BlockId,
}

impl AllocatedBlock {
    const fn new(block: BlockId) -> Self {
        Self { block }
    }

    pub const fn id(&self) -> BlockId {
        self.block
    }
}
