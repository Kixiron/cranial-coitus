use crate::ir::{pretty_utils, Pretty, PrettyConfig};
use pretty::{DocAllocator, DocBuilder};
use std::{
    fmt::{self, Debug, Display, Write},
    ops::{Deref, DerefMut},
};

#[derive(Debug, Clone)]
pub struct Blocks {
    blocks: Vec<BasicBlock>,
}

impl Blocks {
    pub const fn new(blocks: Vec<BasicBlock>) -> Self {
        Self { blocks }
    }
}

impl Deref for Blocks {
    type Target = Vec<BasicBlock>;

    fn deref(&self) -> &Self::Target {
        &self.blocks
    }
}

impl DerefMut for Blocks {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.blocks
    }
}

impl Pretty for Blocks {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator.intersperse(
            self.blocks
                .iter()
                .map(|block| block.pretty(allocator, config)),
            allocator.hardline(),
        )
    }
}

#[derive(Debug, Clone)]
pub struct BasicBlock {
    id: BlockId,
    instructions: Vec<Instruction>,
    terminator: Terminator,
}

impl BasicBlock {
    pub const fn new(id: BlockId, instructions: Vec<Instruction>, terminator: Terminator) -> Self {
        Self {
            id,
            instructions,
            terminator,
        }
    }

    pub const fn id(&self) -> BlockId {
        self.id
    }

    pub fn instructions(&self) -> &[Instruction] {
        self.instructions.as_ref()
    }

    /// Get a mutable reference to the basic block's instructions.
    pub fn instructions_mut(&mut self) -> &mut Vec<Instruction> {
        &mut self.instructions
    }

    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    pub fn push(&mut self, value: Instruction) {
        self.instructions.push(value);
    }

    pub const fn terminator(&self) -> &Terminator {
        &self.terminator
    }

    pub fn terminator_mut(&mut self) -> &mut Terminator {
        &mut self.terminator
    }

    /// Set the basic block's terminator.
    pub fn set_terminator(&mut self, terminator: Terminator) {
        self.terminator = terminator;
    }
}

impl Pretty for BasicBlock {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        self.id
            .pretty(allocator, config)
            .append(allocator.text(":"))
            .append(allocator.hardline())
            .append(
                allocator
                    .intersperse(
                        self.instructions
                            .iter()
                            .map(|inst| inst.pretty(allocator, config)),
                        allocator.hardline(),
                    )
                    .append(if self.instructions.is_empty() {
                        allocator.nil()
                    } else {
                        allocator.hardline()
                    })
                    .append(self.terminator.pretty(allocator, config)),
            )
            .nest(2)
    }
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Store(Store),
    Assign(Assign),
    Output(Output),
}

impl From<Store> for Instruction {
    fn from(store: Store) -> Self {
        Self::Store(store)
    }
}

impl From<Assign> for Instruction {
    fn from(assign: Assign) -> Self {
        Self::Assign(assign)
    }
}

impl From<Output> for Instruction {
    fn from(output: Output) -> Self {
        Self::Output(output)
    }
}

impl Pretty for Instruction {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        match self {
            Self::Store(store) => store.pretty(allocator, config),
            Self::Assign(assign) => assign.pretty(allocator, config),
            Self::Output(output) => output.pretty(allocator, config),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Store {
    ptr: Value,
    value: Value,
}

impl Store {
    pub const fn new(ptr: Value, value: Value) -> Self {
        Self { ptr, value }
    }
}

impl Pretty for Store {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        pretty_utils::binary("store", &self.ptr, &self.value, allocator, config)
    }
}

#[derive(Debug, Clone)]
pub struct Assign {
    pub(super) value: ValId,
    pub(super) rval: RValue,
}

impl Assign {
    pub fn new<R>(value: ValId, rval: R) -> Self
    where
        R: Into<RValue>,
    {
        Self {
            value,
            rval: rval.into(),
        }
    }
}

impl Pretty for Assign {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        self.value
            .pretty(allocator, config)
            .append(allocator.space())
            .append(allocator.text("="))
            .append(allocator.space())
            .append(self.rval.pretty(allocator, config))
    }
}

#[derive(Debug, Clone)]
pub struct Output {
    value: Value,
}

impl Output {
    pub const fn new(value: Value) -> Self {
        Self { value }
    }
}

impl Pretty for Output {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text("call")
            .append(allocator.space())
            .append(allocator.text("output"))
            .append(self.value.pretty(allocator, config).parens())
    }
}

#[derive(Debug, Clone)]
pub enum RValue {
    Eq(Eq),
    Phi(Phi),
    Neg(Neg),
    Not(Not),
    Add(Add),
    Sub(Sub),
    Mul(Mul),
    Load(Load),
    Input(Input),
    BitNot(BitNot),
}

impl From<Phi> for RValue {
    fn from(phi: Phi) -> Self {
        Self::Phi(phi)
    }
}

impl From<Input> for RValue {
    fn from(input: Input) -> Self {
        Self::Input(input)
    }
}

impl From<BitNot> for RValue {
    fn from(bit_not: BitNot) -> Self {
        Self::BitNot(bit_not)
    }
}

impl From<Mul> for RValue {
    fn from(mul: Mul) -> Self {
        Self::Mul(mul)
    }
}

impl From<Sub> for RValue {
    fn from(sub: Sub) -> Self {
        Self::Sub(sub)
    }
}

impl From<Not> for RValue {
    fn from(not: Not) -> Self {
        Self::Not(not)
    }
}

impl From<Neg> for RValue {
    fn from(neg: Neg) -> Self {
        Self::Neg(neg)
    }
}

impl From<Eq> for RValue {
    fn from(eq: Eq) -> Self {
        Self::Eq(eq)
    }
}

impl From<Add> for RValue {
    fn from(add: Add) -> Self {
        Self::Add(add)
    }
}

impl From<Load> for RValue {
    fn from(load: Load) -> Self {
        Self::Load(load)
    }
}

impl Pretty for RValue {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        match self {
            Self::Eq(eq) => eq.pretty(allocator, config),
            Self::Phi(phi) => phi.pretty(allocator, config),
            Self::Neg(neg) => neg.pretty(allocator, config),
            Self::Not(not) => not.pretty(allocator, config),
            Self::Add(add) => add.pretty(allocator, config),
            Self::Sub(sub) => sub.pretty(allocator, config),
            Self::Mul(mul) => mul.pretty(allocator, config),
            Self::Load(load) => load.pretty(allocator, config),
            Self::Input(input) => input.pretty(allocator, config),
            Self::BitNot(bit_not) => bit_not.pretty(allocator, config),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Eq {
    lhs: Value,
    rhs: Value,
}

impl Eq {
    pub const fn new(lhs: Value, rhs: Value) -> Self {
        Self { lhs, rhs }
    }

    pub const fn lhs(&self) -> Value {
        self.lhs
    }

    pub const fn rhs(&self) -> Value {
        self.rhs
    }
}

impl Pretty for Eq {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        pretty_utils::binary("eq", &self.lhs, &self.rhs, allocator, config)
    }
}

#[derive(Debug, Clone)]
pub struct Phi {
    lhs: Value,
    rhs: Value,
}

impl Phi {
    pub const fn new(lhs: Value, rhs: Value) -> Self {
        Self { lhs, rhs }
    }

    pub const fn lhs(&self) -> Value {
        self.lhs
    }

    pub const fn rhs(&self) -> Value {
        self.rhs
    }

    /// Get a mutable reference to the phi's left hand side.
    pub fn lhs_mut(&mut self) -> &mut Value {
        &mut self.lhs
    }

    /// Get a mutable reference to the phi's right hand side.
    pub fn rhs_mut(&mut self) -> &mut Value {
        &mut self.rhs
    }
}

impl Pretty for Phi {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        pretty_utils::binary("phi", &self.lhs, &self.rhs, allocator, config)
    }
}

#[derive(Debug, Clone)]
pub struct Neg {
    value: Value,
}

impl Neg {
    pub const fn new(value: Value) -> Self {
        Self { value }
    }

    pub const fn value(&self) -> Value {
        self.value
    }
}

impl Pretty for Neg {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        pretty_utils::unary("neg", &self.value, allocator, config)
    }
}

#[derive(Debug, Clone)]
pub struct Not {
    value: Value,
}

impl Not {
    pub const fn new(value: Value) -> Self {
        Self { value }
    }

    pub const fn value(&self) -> Value {
        self.value
    }
}

impl Pretty for Not {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        pretty_utils::unary("not", &self.value, allocator, config)
    }
}

#[derive(Debug, Clone)]
pub struct Add {
    lhs: Value,
    rhs: Value,
}

impl Add {
    pub const fn new(lhs: Value, rhs: Value) -> Self {
        Self { lhs, rhs }
    }

    pub const fn lhs(&self) -> Value {
        self.lhs
    }

    pub const fn rhs(&self) -> Value {
        self.rhs
    }
}

impl Pretty for Add {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        pretty_utils::binary("add", &self.lhs, &self.rhs, allocator, config)
    }
}

#[derive(Debug, Clone)]
pub struct Sub {
    lhs: Value,
    rhs: Value,
}

impl Sub {
    pub const fn new(lhs: Value, rhs: Value) -> Self {
        Self { lhs, rhs }
    }

    pub const fn lhs(&self) -> Value {
        self.lhs
    }

    pub const fn rhs(&self) -> Value {
        self.rhs
    }
}

impl Pretty for Sub {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        pretty_utils::binary("sub", &self.lhs, &self.rhs, allocator, config)
    }
}

#[derive(Debug, Clone)]
pub struct Mul {
    lhs: Value,
    rhs: Value,
}

impl Mul {
    pub const fn new(lhs: Value, rhs: Value) -> Self {
        Self { lhs, rhs }
    }

    pub const fn lhs(&self) -> Value {
        self.lhs
    }

    pub const fn rhs(&self) -> Value {
        self.rhs
    }
}

impl Pretty for Mul {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        pretty_utils::binary("mul", &self.lhs, &self.rhs, allocator, config)
    }
}

#[derive(Debug, Clone)]
pub struct Load {
    ptr: Value,
}

impl Load {
    pub const fn new(ptr: Value) -> Self {
        Self { ptr }
    }

    pub const fn ptr(&self) -> Value {
        self.ptr
    }
}

impl Pretty for Load {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        pretty_utils::unary("load", &self.ptr, allocator, config)
    }
}

#[derive(Debug, Clone)]
pub struct Input {}

impl Input {
    pub const fn new() -> Self {
        Self {}
    }
}

impl Pretty for Input {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, _config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator.text("call input()")
    }
}

#[derive(Debug, Clone)]
pub struct BitNot {
    value: Value,
}

impl BitNot {
    pub const fn new(value: Value) -> Self {
        Self { value }
    }

    pub const fn value(&self) -> Value {
        self.value
    }
}

impl Pretty for BitNot {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        pretty_utils::unary("bnot", &self.value, allocator, config)
    }
}

#[derive(Debug, Clone)]
pub enum Terminator {
    Error,
    Unreachable,
    Jump(BlockId),
    Return(Value),
    Branch(Branch),
}

impl Terminator {
    pub const fn as_jump(&self) -> Option<BlockId> {
        if let Self::Jump(target) = *self {
            Some(target)
        } else {
            None
        }
    }

    pub fn as_jump_mut(&mut self) -> Option<&mut BlockId> {
        if let Self::Jump(target) = self {
            Some(target)
        } else {
            None
        }
    }

    pub fn as_branch_mut(&mut self) -> Option<&mut Branch> {
        if let Self::Branch(branch) = self {
            Some(branch)
        } else {
            None
        }
    }
}

impl Pretty for Terminator {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        match self {
            Self::Error => allocator.text("error"),
            Self::Unreachable => allocator.text("unreachable"),

            Self::Jump(dest) => allocator
                .text("jmp")
                .append(allocator.space())
                .append(dest.pretty(allocator, config)),

            Self::Return(value) => allocator
                .text("ret")
                .append(allocator.space())
                .append(value.pretty(allocator, config)),

            Self::Branch(branch) => allocator
                .text("br")
                .append(allocator.space())
                .append(branch.condition.pretty(allocator, config))
                .append(allocator.text(","))
                .append(allocator.space())
                .append(
                    branch
                        .true_jump
                        .pretty(allocator, config)
                        .append(allocator.text(","))
                        .append(allocator.space())
                        .append(branch.false_jump.pretty(allocator, config))
                        .brackets(),
                ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Branch {
    condition: Value,
    true_jump: BlockId,
    false_jump: BlockId,
}

impl Branch {
    pub const fn new(condition: Value, true_jump: BlockId, false_jump: BlockId) -> Self {
        Self {
            condition,
            true_jump,
            false_jump,
        }
    }

    pub const fn condition(&self) -> Value {
        self.condition
    }

    pub const fn true_jump(&self) -> BlockId {
        self.true_jump
    }

    /// Set the branch's true jump.
    pub fn set_true_jump(&mut self, true_jump: BlockId) {
        self.true_jump = true_jump;
    }

    pub const fn false_jump(&self) -> BlockId {
        self.false_jump
    }

    /// Set the branch's false jump.
    pub fn set_false_jump(&mut self, false_jump: BlockId) {
        self.false_jump = false_jump;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Value {
    Byte(u8),
    Uint(u32),
    Bool(bool),
    Val(ValId),
}

impl From<ValId> for Value {
    fn from(val: ValId) -> Self {
        Self::Val(val)
    }
}

impl Pretty for Value {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        match self {
            Self::Byte(byte) => allocator
                .text("byte")
                .append(allocator.space())
                .append(allocator.text(format!("{}", byte))),

            Self::Uint(uint) => allocator
                .text("uint")
                .append(allocator.space())
                .append(allocator.text(format!("{}", uint))),

            Self::Bool(bool) => allocator
                .text("bool")
                .append(allocator.space())
                .append(allocator.text(format!("{}", bool))),

            Self::Val(val) => val.pretty(allocator, config),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct ValId(pub u32);

impl ValId {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
}

impl Debug for ValId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ValId(")?;
        Debug::fmt(&self.0, f)?;
        f.write_char(')')
    }
}

impl Display for ValId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_char('%')?;
        Display::fmt(&self.0, f)
    }
}

impl Pretty for ValId {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, _config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator.text(format!("{}", self))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct BlockId(pub u32);

impl BlockId {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
}

impl Debug for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BlockId(")?;
        Debug::fmt(&self.0, f)?;
        f.write_char(')')
    }
}

impl Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("bb")?;
        Display::fmt(&self.0, f)
    }
}

impl Pretty for BlockId {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, _config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator.text(format!("{}", self))
    }
}
