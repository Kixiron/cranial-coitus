use crate::{
    ir::{pretty_utils, CmpKind, Pretty, PrettyConfig},
    utils::{DebugDisplay, HashMap},
    values::{Cell, Ptr},
};
use petgraph::{graph::NodeIndex, visit::EdgeRef, Direction, Graph};
use pretty::{DocAllocator, DocBuilder};
use std::{
    fmt::{self, Debug, Display, Write},
    num::Wrapping,
    ops::{Deref, DerefMut},
    slice, vec,
};

const COMMENT_ALIGNMENT_OFFSET: usize = 20;

#[derive(Debug, Clone)]
pub struct Blocks {
    entry: BlockId,
    blocks: Vec<BasicBlock>,
    cfg: Graph<BlockId, ()>,
    nodes: HashMap<BlockId, NodeIndex>,
}

impl Blocks {
    pub const fn new(
        entry: BlockId,
        blocks: Vec<BasicBlock>,
        cfg: Graph<BlockId, ()>,
        nodes: HashMap<BlockId, NodeIndex>,
    ) -> Self {
        Self {
            entry,
            blocks,
            cfg,
            nodes,
        }
    }

    pub fn entry(&self) -> &BasicBlock {
        self.blocks
            .iter()
            .find(|block| block.id() == self.entry)
            .unwrap()
    }

    pub fn incoming_jumps(&self, block: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        self.cfg
            .edges_directed(self.nodes[&block], Direction::Incoming)
            .into_iter()
            .map(|edge| self.cfg[edge.source()])
    }

    pub fn outgoing_jumps(&self, block: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        self.cfg
            .edges_directed(self.nodes[&block], Direction::Outgoing)
            .into_iter()
            .map(|edge| self.cfg[edge.target()])
    }

    pub fn as_slice(&self) -> &[BasicBlock] {
        self.blocks.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [BasicBlock] {
        self.blocks.as_mut_slice()
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

impl<'a> IntoIterator for &'a Blocks {
    type Item = &'a BasicBlock;
    type IntoIter = slice::Iter<'a, BasicBlock>;

    fn into_iter(self) -> Self::IntoIter {
        self.blocks.as_slice().iter()
    }
}

impl<'a> IntoIterator for &'a mut Blocks {
    type Item = &'a mut BasicBlock;
    type IntoIter = slice::IterMut<'a, BasicBlock>;

    fn into_iter(self) -> Self::IntoIter {
        self.blocks.as_mut_slice().iter_mut()
    }
}

impl IntoIterator for Blocks {
    type Item = BasicBlock;
    type IntoIter = vec::IntoIter<BasicBlock>;

    fn into_iter(self) -> Self::IntoIter {
        self.blocks.into_iter()
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
            allocator.hardline().append(allocator.hardline()),
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

    pub fn as_slice(&self) -> &[Instruction] {
        self.instructions.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [Instruction] {
        self.instructions.as_mut_slice()
    }

    pub fn iter(&self) -> slice::Iter<'_, Instruction> {
        self.as_slice().iter()
    }

    pub fn iter_mut(&mut self) -> slice::IterMut<'_, Instruction> {
        self.as_mut_slice().iter_mut()
    }
}

impl<'a> IntoIterator for &'a BasicBlock {
    type Item = &'a Instruction;
    type IntoIter = slice::Iter<'a, Instruction>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut BasicBlock {
    type Item = &'a mut Instruction;
    type IntoIter = slice::IterMut<'a, Instruction>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl IntoIterator for BasicBlock {
    type Item = Instruction;
    type IntoIter = vec::IntoIter<Instruction>;

    fn into_iter(self) -> Self::IntoIter {
        self.instructions.into_iter()
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

impl Instruction {
    pub const fn as_assign(&self) -> Option<&Assign> {
        if let Self::Assign(assign) = self {
            Some(assign)
        } else {
            None
        }
    }

    pub fn as_mut_assign(&mut self) -> Option<&mut Assign> {
        if let Self::Assign(assign) = self {
            Some(assign)
        } else {
            None
        }
    }
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

    pub fn ptr(&self) -> Value {
        self.ptr
    }

    pub fn value(&self) -> Value {
        self.value
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

/// An assignment instruction that assigns an evaluated value to a name
#[derive(Debug, Clone)]
pub struct Assign {
    value: ValId,
    rval: RValue,
}

impl Assign {
    /// Create a new assignment
    pub fn new<R>(value: ValId, rval: R) -> Self
    where
        R: Into<RValue>,
    {
        Self {
            value,
            rval: rval.into(),
        }
    }

    /// Get the value this assignment assigns to
    pub const fn value(&self) -> ValId {
        self.value
    }

    /// Get a reference to the assignment's right value.
    pub const fn rval(&self) -> &RValue {
        &self.rval
    }

    /// Get a mutable reference to the assignment's right value.
    pub fn rval_mut(&mut self) -> &mut RValue {
        &mut self.rval
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

    pub fn value(&self) -> Value {
        self.value
    }
}

impl Pretty for Output {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator.column(move |start_column| {
            allocator
                .text("call")
                .append(allocator.space())
                .append(allocator.text("output"))
                .append(self.value.pretty(allocator, config).parens())
                .append(allocator.column(move |column| {
                    let char = match self.value() {
                        Value::U8(byte) => byte.into_inner() as char,
                        Value::U16(long) => {
                            if let Some(char) = char::from_u32(long.0 as u32) {
                                char
                            } else {
                                return allocator.nil().into_doc();
                            }
                        }
                        Value::TapePtr(ptr) => {
                            if let Some(char) = char::from_u32(ptr.value() as u32) {
                                char
                            } else {
                                return allocator.nil().into_doc();
                            }
                        }
                        Value::Bool(bool) => bool as u8 as char,
                        Value::Val(..) => return allocator.nil().into_doc(),
                    };
                    let comment = format!("// {:?}", char);

                    allocator
                        .space()
                        .append(
                            allocator.text(comment).indent(
                                COMMENT_ALIGNMENT_OFFSET.saturating_sub(column - start_column),
                            ),
                        )
                        .into_doc()
                }))
                .into_doc()
        })
    }
}

#[derive(Debug, Clone)]
pub enum RValue {
    Cmp(Cmp),
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

impl RValue {
    pub const fn as_phi(&self) -> Option<&Phi> {
        if let Self::Phi(phi) = self {
            Some(phi)
        } else {
            None
        }
    }

    pub fn as_mut_phi(&mut self) -> Option<&mut Phi> {
        if let Self::Phi(phi) = self {
            Some(phi)
        } else {
            None
        }
    }
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

impl From<Cmp> for RValue {
    fn from(eq: Cmp) -> Self {
        Self::Cmp(eq)
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
            Self::Cmp(eq) => eq.pretty(allocator, config),
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
pub struct Cmp {
    lhs: Value,
    rhs: Value,
    kind: CmpKind,
}

impl Cmp {
    pub const fn new(lhs: Value, rhs: Value, kind: CmpKind) -> Self {
        Self { lhs, rhs, kind }
    }

    pub const fn lhs(&self) -> Value {
        self.lhs
    }

    pub const fn rhs(&self) -> Value {
        self.rhs
    }

    pub const fn kind(&self) -> CmpKind {
        self.kind
    }
}

impl Pretty for Cmp {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text("cmp.")
            .append(self.kind.pretty(allocator, config))
            .append(allocator.space())
            .append(self.lhs.pretty(allocator, config))
            .append(allocator.text(","))
            .append(allocator.space())
            .append(self.rhs.pretty(allocator, config))
    }
}

#[derive(Debug, Clone)]
pub struct Phi {
    lhs: Value,
    lhs_src: BlockId,
    rhs: Value,
    rhs_src: BlockId,
}

impl Phi {
    pub const fn new(lhs: Value, lhs_src: BlockId, rhs: Value, rhs_src: BlockId) -> Self {
        Self {
            lhs,
            lhs_src,
            rhs,
            rhs_src,
        }
    }

    pub const fn lhs(&self) -> Value {
        self.lhs
    }

    pub const fn lhs_src(&self) -> BlockId {
        self.lhs_src
    }

    pub const fn rhs(&self) -> Value {
        self.rhs
    }

    pub const fn rhs_src(&self) -> BlockId {
        self.rhs_src
    }

    /// Get a mutable reference to the phi's left hand side.
    pub fn lhs_mut(&mut self) -> &mut Value {
        &mut self.lhs
    }

    pub fn lhs_src_mut(&mut self) -> &mut BlockId {
        &mut self.lhs_src
    }

    /// Get a mutable reference to the phi's right hand side.
    pub fn rhs_mut(&mut self) -> &mut Value {
        &mut self.rhs
    }

    pub fn rhs_src_mut(&mut self) -> &mut BlockId {
        &mut self.rhs_src
    }
}

impl Pretty for Phi {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        let phi_val = |value: &'a Value, src: &'a BlockId| {
            value
                .pretty(allocator, config)
                .append(allocator.text(","))
                .append(allocator.space())
                .append(src.pretty(allocator, config))
                .brackets()
        };

        allocator
            .text("phi")
            .append(allocator.space())
            .append(phi_val(&self.lhs, &self.lhs_src))
            .append(allocator.text(","))
            .append(allocator.space())
            .append(phi_val(&self.rhs, &self.rhs_src))
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

    pub fn as_mut_jump(&mut self) -> Option<&mut BlockId> {
        if let Self::Jump(target) = self {
            Some(target)
        } else {
            None
        }
    }

    pub fn as_mut_branch(&mut self) -> Option<&mut Branch> {
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
    U8(Cell),
    U16(Wrapping<u16>),
    TapePtr(Ptr),
    Bool(bool),
    Val(ValId, Type),
}

impl Value {
    pub const fn val(value: ValId, ty: Type) -> Self {
        Self::Val(value, ty)
    }

    pub const fn ty(self) -> Type {
        match self {
            Self::U8(_) => Type::U8,
            Self::U16(_) => Type::U16,
            Self::TapePtr(_) => Type::Ptr,
            Self::Bool(_) => Type::Bool,
            Self::Val(_, ty) => ty,
        }
    }
}

impl From<Cell> for Value {
    fn from(u8: Cell) -> Self {
        Self::U8(u8)
    }
}

impl From<(ValId, Type)> for Value {
    fn from((value, ty): (ValId, Type)) -> Self {
        Self::val(value, ty)
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
            Self::U8(u8) => allocator
                .text("u8")
                .append(allocator.space())
                .append(allocator.text(format!("{}", u8))),

            Self::U16(u16) => allocator
                .text("u16")
                .append(allocator.space())
                .append(allocator.text(format!("{}", u16))),

            Self::TapePtr(ptr) => allocator
                .text("ptr")
                .append(allocator.space())
                .append(allocator.text(format!("{}", ptr))),

            Self::Bool(bool) => allocator
                .text("bool")
                .append(allocator.space())
                .append(allocator.text(format!("{}", bool))),

            Self::Val(value, ty) => ty
                .pretty(allocator, config)
                .append(allocator.space())
                .append(value.pretty(allocator, config)),
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

    pub fn display(&self) -> DebugDisplay<Self> {
        DebugDisplay::new(*self)
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
        f.write_char('v')?;
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

    pub fn display(&self) -> DebugDisplay<Self> {
        DebugDisplay::new(*self)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Type {
    /// A single `u8`
    U8,
    /// A single `u16`, unbounded with wrapping arithmetic
    U16,
    /// A boolean value
    Bool,
    /// A tape pointer, bounded from `0..tape_len`
    Ptr,
}

impl Type {
    /// Returns `true` if the type is a [`Ptr`].
    ///
    /// [`Ptr`]: Type::Ptr
    pub const fn is_ptr(&self) -> bool {
        matches!(self, Self::Ptr)
    }
}

impl Pretty for Type {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, _config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        match self {
            Self::U8 => allocator.text("u8"),
            Self::U16 => allocator.text("u16"),
            Self::Bool => allocator.text("bool"),
            Self::Ptr => allocator.text("ptr"),
        }
    }
}
