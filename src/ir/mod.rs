mod builder;

pub use builder::IrBuilder;

use pretty::{Arena, DocAllocator, DocBuilder};
use std::{
    borrow::Cow,
    fmt::{self, Debug, Display, Write},
};

pub trait Pretty {
    fn pretty_print(&self) -> String {
        let arena = Arena::<()>::new();
        self.pretty(&arena).1.pretty(80).to_string()
    }

    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone;
}

#[derive(Debug, Clone)]
pub struct Block {
    instructions: Vec<Instruction>,
}

impl Block {
    pub fn new(instructions: Vec<Instruction>) -> Self {
        Self { instructions }
    }

    pub fn into_inner(self) -> Vec<Instruction> {
        self.instructions
    }
}

impl Pretty for Block {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .intersperse(
                self.instructions.iter().map(|inst| inst.pretty(allocator)),
                allocator.hardline(),
            )
            .append(allocator.hardline())
    }
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Call(Call),
    Assign(Assign),
    Theta(Theta),
    Phi(Phi),
    Store(Store),
}

impl Pretty for Instruction {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        match self {
            Self::Call(call) => call.pretty(allocator),
            Self::Assign(assign) => assign.pretty(allocator),
            Self::Theta(theta) => theta.pretty(allocator),
            Self::Phi(phi) => phi.pretty(allocator),
            Self::Store(store) => store.pretty(allocator),
        }
    }
}

impl From<Store> for Instruction {
    fn from(store: Store) -> Self {
        Self::Store(store)
    }
}

impl From<Theta> for Instruction {
    fn from(theta: Theta) -> Self {
        Self::Theta(theta)
    }
}

impl From<Phi> for Instruction {
    fn from(phi: Phi) -> Self {
        Self::Phi(phi)
    }
}

impl From<Assign> for Instruction {
    fn from(assign: Assign) -> Self {
        Self::Assign(assign)
    }
}

impl From<Call> for Instruction {
    fn from(call: Call) -> Self {
        Self::Call(call)
    }
}

#[derive(Debug, Clone)]
pub struct Theta {
    body: Vec<Instruction>,
    cond: Option<Value>,
    pred_effect: Option<VarId>,
}

impl Theta {
    pub fn new<C, E>(body: Vec<Instruction>, cond: C, pred_effect: E) -> Self
    where
        C: Into<Option<Value>>,
        E: Into<Option<VarId>>,
    {
        Self {
            body,
            cond: cond.into(),
            pred_effect: pred_effect.into(),
        }
    }
}

impl Pretty for Theta {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text(if let Some(effect) = self.pred_effect {
                Cow::Owned(format!("// pred: {}", effect))
            } else {
                Cow::Borrowed("// pred: ???")
            })
            .append(allocator.hardline())
            .append(allocator.text("do"))
            .append(allocator.space())
            .append(allocator.text("{"))
            .append(if self.body.is_empty() {
                allocator.nil()
            } else {
                allocator.hardline()
            })
            .append(
                allocator
                    .intersperse(
                        self.body.iter().map(|inst| inst.pretty(allocator)),
                        allocator.hardline(),
                    )
                    .indent(2),
            )
            .append(if self.body.is_empty() {
                allocator.nil()
            } else {
                allocator.hardline()
            })
            .append(allocator.text("}"))
            .append(allocator.space())
            .append(allocator.text("while"))
            .append(allocator.space())
            .append(allocator.text("{"))
            .append(allocator.space())
            .append(if let Some(cond) = self.cond.as_ref() {
                cond.pretty(allocator)
            } else {
                allocator.text("???")
            })
            .append(allocator.space())
            .append(allocator.text("}"))
    }
}

#[derive(Debug, Clone)]
pub struct Phi {
    cond: Value,
    truthy: Vec<Instruction>,
    falsy: Vec<Instruction>,
    pred_effect: Option<VarId>,
}

impl Phi {
    pub fn new<C, E>(
        cond: C,
        truthy: Vec<Instruction>,
        falsy: Vec<Instruction>,
        pred_effect: E,
    ) -> Self
    where
        C: Into<Value>,
        E: Into<Option<VarId>>,
    {
        Self {
            cond: cond.into(),
            truthy,
            falsy,
            pred_effect: pred_effect.into(),
        }
    }
}

impl Pretty for Phi {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text(if let Some(effect) = self.pred_effect {
                Cow::Owned(format!("// pred: {}", effect))
            } else {
                Cow::Borrowed("// pred: ???")
            })
            .append(allocator.hardline())
            .append(allocator.text("if"))
            .append(allocator.space())
            .append(self.cond.pretty(allocator))
            .append(allocator.space())
            .append(allocator.text("{"))
            .append(allocator.hardline())
            .append(
                allocator
                    .intersperse(
                        self.truthy.iter().map(|inst| inst.pretty(allocator)),
                        allocator.hardline(),
                    )
                    .indent(2),
            )
            .append(allocator.hardline())
            .append(allocator.text("}"))
            .append(allocator.space())
            .append(allocator.text("else"))
            .append(allocator.space())
            .append(allocator.text("{"))
            .append(allocator.hardline())
            .append(
                allocator
                    .intersperse(
                        self.falsy.iter().map(|inst| inst.pretty(allocator)),
                        allocator.hardline(),
                    )
                    .indent(2),
            )
            .append(allocator.hardline())
            .append(allocator.text("}"))
    }
}

#[derive(Debug, Clone)]
pub struct Call {
    function: Cow<'static, str>,
    args: Vec<Value>,
    pred_effect: Option<VarId>,
}

impl Call {
    pub fn new<F, E>(function: F, args: Vec<Value>, pred_effect: E) -> Self
    where
        F: Into<Cow<'static, str>>,
        E: Into<Option<VarId>>,
    {
        Self {
            function: function.into(),
            args,
            pred_effect: pred_effect.into(),
        }
    }
}

impl Pretty for Call {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text("call")
            .append(allocator.space())
            .append(
                allocator.text(self.function.clone()).append(
                    allocator
                        .intersperse(
                            self.args.iter().map(|arg| arg.pretty(allocator)),
                            allocator.text(","),
                        )
                        .parens(),
                ),
            )
            .append(allocator.space())
            .append(allocator.text(if let Some(effect) = self.pred_effect {
                Cow::Owned(format!("// pred: {}", effect))
            } else {
                Cow::Borrowed("// pred: ???")
            }))
    }
}

#[derive(Debug, Clone)]
pub struct Assign {
    var: VarId,
    value: Expr,
}

impl Assign {
    pub fn new<I>(var: VarId, value: I) -> Self
    where
        I: Into<Expr>,
    {
        Self {
            var,
            value: value.into(),
        }
    }
}

impl Pretty for Assign {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        self.var
            .pretty(allocator)
            .append(allocator.space())
            .append(allocator.text(":="))
            .append(allocator.space())
            .append(self.value.pretty(allocator))
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    Eq(Eq),
    Add(Add),
    Not(Not),
    Const(Const),
    Load(Load),
    Call(Call),
}

impl Pretty for Expr {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        match self {
            Self::Eq(eq) => eq.pretty(allocator),
            Self::Add(add) => add.pretty(allocator),
            Self::Not(not) => not.pretty(allocator),
            Self::Const(constant) => constant.pretty(allocator),
            Self::Load(load) => load.pretty(allocator),
            Self::Call(call) => call.pretty(allocator),
        }
    }
}

impl From<Call> for Expr {
    fn from(v: Call) -> Self {
        Self::Call(v)
    }
}

impl From<Load> for Expr {
    fn from(v: Load) -> Self {
        Self::Load(v)
    }
}

impl From<Not> for Expr {
    fn from(v: Not) -> Self {
        Self::Not(v)
    }
}

impl From<Eq> for Expr {
    fn from(eq: Eq) -> Self {
        Self::Eq(eq)
    }
}

impl From<Add> for Expr {
    fn from(add: Add) -> Self {
        Self::Add(add)
    }
}

impl From<Const> for Expr {
    fn from(value: Const) -> Self {
        Self::Const(value)
    }
}

#[derive(Debug, Clone)]
pub struct Add {
    lhs: Value,
    rhs: Value,
}

impl Add {
    pub fn new<L, R>(lhs: L, rhs: R) -> Self
    where
        L: Into<Value>,
        R: Into<Value>,
    {
        Self {
            lhs: lhs.into(),
            rhs: rhs.into(),
        }
    }
}

impl Pretty for Add {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text("add")
            .append(allocator.space())
            .append(self.lhs.pretty(allocator))
            .append(allocator.text(","))
            .append(allocator.space())
            .append(self.rhs.pretty(allocator))
    }
}

#[derive(Debug, Clone)]
pub struct Not {
    value: Value,
}

impl Not {
    pub fn new<V>(value: V) -> Self
    where
        V: Into<Value>,
    {
        Self {
            value: value.into(),
        }
    }
}

impl Pretty for Not {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text("not")
            .append(allocator.space())
            .append(self.value.pretty(allocator))
    }
}

#[derive(Debug, Clone)]
pub struct Eq {
    lhs: Value,
    rhs: Value,
}

impl Eq {
    pub fn new<L, R>(lhs: L, rhs: R) -> Self
    where
        L: Into<Value>,
        R: Into<Value>,
    {
        Self {
            lhs: lhs.into(),
            rhs: rhs.into(),
        }
    }
}

impl Pretty for Eq {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text("eq")
            .append(allocator.space())
            .append(self.lhs.pretty(allocator))
            .append(allocator.text(","))
            .append(allocator.space())
            .append(self.rhs.pretty(allocator))
    }
}

#[derive(Debug, Clone)]
pub struct Load {
    ptr: Value,
    pred_effect: Option<VarId>,
}

impl Load {
    pub fn new<E>(ptr: Value, pred_effect: E) -> Self
    where
        E: Into<Option<VarId>>,
    {
        Self {
            ptr,
            pred_effect: pred_effect.into(),
        }
    }
}

impl Pretty for Load {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text("load")
            .append(allocator.space())
            .append(self.ptr.pretty(allocator))
            .append(allocator.space())
            .append(allocator.text(if let Some(effect) = self.pred_effect {
                Cow::Owned(format!("// pred: {}", effect))
            } else {
                Cow::Borrowed("// pred: ???")
            }))
    }
}

#[derive(Debug, Clone)]
pub struct Store {
    ptr: Value,
    value: Value,
    pred_effect: Option<VarId>,
}

impl Store {
    pub fn new<E>(ptr: Value, value: Value, pred_effect: E) -> Self
    where
        E: Into<Option<VarId>>,
    {
        Self {
            ptr,
            value,
            pred_effect: pred_effect.into(),
        }
    }
}

impl Pretty for Store {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text("store")
            .append(allocator.space())
            .append(self.ptr.pretty(allocator))
            .append(allocator.text(","))
            .append(allocator.space())
            .append(self.value.pretty(allocator))
            .append(allocator.space())
            .append(allocator.text(if let Some(effect) = self.pred_effect {
                Cow::Owned(format!("// pred: {}", effect))
            } else {
                Cow::Borrowed("// pred: ???")
            }))
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    Var(VarId),
    Const(Const),
}

impl Value {
    pub fn as_var(&self) -> Option<VarId> {
        if let Self::Var(var) = *self {
            Some(var)
        } else {
            None
        }
    }
}

impl Pretty for Value {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        match self {
            Self::Var(var) => var.pretty(allocator),
            Self::Const(constant) => constant.pretty(allocator),
        }
    }
}

impl From<VarId> for Value {
    fn from(var: VarId) -> Self {
        Self::Var(var)
    }
}

impl From<Const> for Value {
    fn from(value: Const) -> Self {
        Self::Const(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Const {
    Int(i32),
    Bool(bool),
}

impl Const {
    pub fn convert_to_i32(&self) -> Option<i32> {
        match *self {
            Self::Int(int) => Some(int),
            Self::Bool(bool) => Some(bool as i32),
        }
    }

    // pub fn convert_to_u8(&self) -> Option<u8> {
    //     match *self {
    //         Self::Int(int) => Some(int.rem_euclid(u8::MAX as i32) as u8),
    //         Self::Bool(bool) => Some(bool as u8),
    //     }
    // }

    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(bool) = *self {
            Some(bool)
        } else {
            None
        }
    }

    pub fn as_int(&self) -> Option<i32> {
        if let Self::Int(int) = *self {
            Some(int)
        } else {
            None
        }
    }
}

impl Pretty for Const {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        let text = match *self {
            Self::Int(int) => format!("int {}", int),
            Self::Bool(boolean) => format!("bool {}", boolean),
        };
        allocator.text(text)
    }
}

impl From<i32> for Const {
    fn from(int: i32) -> Self {
        Self::Int(int)
    }
}

impl From<bool> for Const {
    fn from(boolean: bool) -> Self {
        Self::Bool(boolean)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct VarId(u32);

impl VarId {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
}

impl Pretty for VarId {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator.text(format!("_{}", self))
    }
}

impl Debug for VarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("VarId(")?;
        Debug::fmt(&self.0, f)?;
        f.write_char(')')
    }
}

impl Display for VarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}
