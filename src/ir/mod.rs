mod builder;

pub use builder::IrBuilder;

use pretty::{Arena, DocAllocator, DocBuilder};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    fmt::{self, Debug, Display, Write},
    ops::{self, Deref, DerefMut},
    time::Instant,
};

use crate::graph::{OutputPort, Port};

pub trait Pretty {
    fn pretty_print(&self) -> String {
        let start_time = Instant::now();

        let arena = Arena::<()>::new();
        let pretty = self.pretty(&arena).1.pretty(80).to_string();

        let elapsed = start_time.elapsed();
        tracing::debug!(
            target: "timings",
            "took {:#?} to pretty print ir",
            elapsed,
        );

        pretty
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
    pub const fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            instructions: Vec::with_capacity(capacity),
        }
    }

    pub fn into_inner(self) -> Vec<Instruction> {
        self.instructions
    }
}

impl Deref for Block {
    type Target = Vec<Instruction>;

    fn deref(&self) -> &Self::Target {
        &self.instructions
    }
}

impl DerefMut for Block {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.instructions
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
    Gamma(Gamma),
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
            Self::Gamma(gamma) => gamma.pretty(allocator),
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

impl From<Gamma> for Instruction {
    fn from(gamma: Gamma) -> Self {
        Self::Gamma(gamma)
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
    pub body: Vec<Instruction>,
    pub cond: Option<Value>,
    pub output_effect: Option<EffectId>,
    pub input_effect: Option<EffectId>,
    pub inputs: BTreeMap<VarId, Value>,
    pub outputs: BTreeMap<VarId, Value>,
    pub output_feedback: BTreeMap<VarId, VarId>,
}

impl Theta {
    pub fn new<C, E1, E2>(
        body: Vec<Instruction>,
        cond: C,
        output_effect: E1,
        input_effect: E2,
        inputs: BTreeMap<VarId, Value>,
        outputs: BTreeMap<VarId, Value>,
        output_feedback: BTreeMap<VarId, VarId>,
    ) -> Self
    where
        C: Into<Option<Value>>,
        E1: Into<Option<EffectId>>,
        E2: Into<Option<EffectId>>,
    {
        Self {
            body,
            cond: cond.into(),
            output_effect: output_effect.into(),
            input_effect: input_effect.into(),
            inputs,
            outputs,
            output_feedback,
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
            .text(match (self.input_effect, self.output_effect) {
                (None, None) => "// eff: ???, pred: ???".to_owned(),
                (None, Some(output_effect)) => format!("// eff: {}, pred: ???", output_effect),
                (Some(input_effect), None) => format!("// eff: ???, pred: {}", input_effect),
                (Some(input_effect), Some(output_effect)) => {
                    format!("// eff: {}, pred: {}", output_effect, input_effect)
                }
            })
            .append(allocator.hardline())
            .append(allocator.text("do"))
            .append(allocator.space())
            .append(allocator.text("{"))
            .append(if self.body.is_empty() {
                allocator.nil()
            } else {
                allocator
                    .hardline()
                    .append(
                        allocator
                            .intersperse(
                                self.body.iter().map(|inst| inst.pretty(allocator)),
                                allocator.hardline(),
                            )
                            .indent(2),
                    )
                    .append(allocator.hardline())
            })
            .append(if self.outputs.is_empty() {
                allocator.nil()
            } else {
                if self.body.is_empty() {
                    allocator.hardline()
                } else {
                    allocator.nil()
                }
                .append(
                    allocator
                        .text("// outputs: ")
                        .append(allocator.intersperse(
                            self.outputs.iter().map(|(var, value)| {
                                var.pretty(allocator)
                                    .append(allocator.text(" = "))
                                    .append(value.pretty(allocator))
                            }),
                            allocator.text(", "),
                        ))
                        .indent(2),
                )
                .append(allocator.hardline())
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
pub struct Gamma {
    pub cond: Value,
    pub truthy: Vec<Instruction>,
    pub true_outputs: BTreeMap<VarId, Value>,
    pub falsy: Vec<Instruction>,
    pub false_outputs: BTreeMap<VarId, Value>,
    pub effect: EffectId,
    pub prev_effect: Option<EffectId>,
}

impl Gamma {
    pub fn new<C, E>(
        cond: C,
        truthy: Vec<Instruction>,
        true_outputs: BTreeMap<VarId, Value>,
        falsy: Vec<Instruction>,
        false_outputs: BTreeMap<VarId, Value>,
        effect: EffectId,
        prev_effect: E,
    ) -> Self
    where
        C: Into<Value>,
        E: Into<Option<EffectId>>,
    {
        Self {
            cond: cond.into(),
            truthy,
            true_outputs,
            falsy,
            false_outputs,
            effect,
            prev_effect: prev_effect.into(),
        }
    }
}

impl Pretty for Gamma {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text(if let Some(prev_effect) = self.prev_effect {
                format!("// eff: {}, pred: {}", self.effect, prev_effect)
            } else {
                format!("// eff: {}, pred: ???", self.effect)
            })
            .append(allocator.hardline())
            .append(allocator.text("if"))
            .append(allocator.space())
            .append(self.cond.pretty(allocator))
            .append(allocator.space())
            .append(allocator.text("{"))
            .append(if self.truthy.is_empty() {
                allocator.nil()
            } else {
                allocator
                    .hardline()
                    .append(
                        allocator
                            .intersperse(
                                self.truthy.iter().map(|inst| inst.pretty(allocator)),
                                allocator.hardline(),
                            )
                            .indent(2),
                    )
                    .append(allocator.hardline())
            })
            .append(if self.true_outputs.is_empty() {
                allocator.nil()
            } else {
                if self.truthy.is_empty() {
                    allocator.hardline()
                } else {
                    allocator.nil()
                }
                .append(
                    allocator
                        .text("// outputs: ")
                        .append(allocator.intersperse(
                            self.true_outputs.iter().map(|(var, value)| {
                                var.pretty(allocator)
                                    .append(allocator.text(" = "))
                                    .append(value.pretty(allocator))
                            }),
                            allocator.text(", "),
                        ))
                        .indent(2),
                )
                .append(allocator.hardline())
            })
            .append(allocator.text("}"))
            .append(allocator.space())
            .append(allocator.text("else"))
            .append(allocator.space())
            .append(allocator.text("{"))
            .append(if self.falsy.is_empty() {
                allocator.nil()
            } else {
                allocator
                    .hardline()
                    .append(
                        allocator
                            .intersperse(
                                self.falsy.iter().map(|inst| inst.pretty(allocator)),
                                allocator.hardline(),
                            )
                            .indent(2),
                    )
                    .append(allocator.hardline())
            })
            .append(if self.false_outputs.is_empty() {
                allocator.nil()
            } else {
                if self.falsy.is_empty() {
                    allocator.hardline()
                } else {
                    allocator.nil()
                }
                .append(
                    allocator
                        .text("// outputs: ")
                        .append(allocator.intersperse(
                            self.false_outputs.iter().map(|(var, value)| {
                                var.pretty(allocator)
                                    .append(allocator.text(" = "))
                                    .append(value.pretty(allocator))
                            }),
                            allocator.text(", "),
                        ))
                        .indent(2),
                )
                .append(allocator.hardline())
            })
            .append(allocator.text("}"))
    }
}

#[derive(Debug, Clone)]
pub struct Call {
    pub function: Cow<'static, str>,
    pub args: Vec<Value>,
    pub effect: EffectId,
    pub prev_effect: Option<EffectId>,
}

impl Call {
    pub fn new<F, E>(function: F, args: Vec<Value>, effect: EffectId, prev_effect: E) -> Self
    where
        F: Into<Cow<'static, str>>,
        E: Into<Option<EffectId>>,
    {
        Self {
            function: function.into(),
            args,
            effect,
            prev_effect: prev_effect.into(),
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
            .append(allocator.text(if let Some(prev_effect) = self.prev_effect {
                format!("// eff: {}, pred: {}", self.effect, prev_effect)
            } else {
                format!("// eff: {}, pred: ???", self.effect)
            }))
    }
}

#[derive(Debug, Clone)]
pub struct Assign {
    pub var: VarId,
    pub value: Expr,
    pub tag: AssignTag,
}

impl Assign {
    pub fn new<I>(var: VarId, value: I) -> Self
    where
        I: Into<Expr>,
    {
        Self {
            var,
            value: value.into(),
            tag: AssignTag::None,
        }
    }

    pub fn input<I>(var: VarId, value: I) -> Self
    where
        I: Into<Expr>,
    {
        Self {
            var,
            value: value.into(),
            tag: AssignTag::InputParam,
        }
    }

    pub fn output<I>(var: VarId, value: I) -> Self
    where
        I: Into<Expr>,
    {
        Self {
            var,
            value: value.into(),
            tag: AssignTag::OutputParam,
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
            .append(match self.tag {
                AssignTag::None => allocator.nil(),
                AssignTag::InputParam => allocator.text("in").append(allocator.space()),
                AssignTag::OutputParam => allocator.text("out").append(allocator.space()),
            })
            .append(self.value.pretty(allocator))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssignTag {
    None,
    InputParam,
    OutputParam,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Eq(Eq),
    Add(Add),
    Not(Not),
    Neg(Neg),
    Load(Load),
    Call(Call),
    Value(Value),
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
            Self::Neg(neg) => neg.pretty(allocator),
            Self::Load(load) => load.pretty(allocator),
            Self::Call(call) => call.pretty(allocator),
            Self::Value(value) => value.pretty(allocator),
        }
    }
}

impl From<Call> for Expr {
    fn from(call: Call) -> Self {
        Self::Call(call)
    }
}

impl From<Load> for Expr {
    fn from(load: Load) -> Self {
        Self::Load(load)
    }
}

impl From<Not> for Expr {
    fn from(not: Not) -> Self {
        Self::Not(not)
    }
}

impl From<Neg> for Expr {
    fn from(neg: Neg) -> Self {
        Self::Neg(neg)
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

impl From<Value> for Expr {
    fn from(value: Value) -> Self {
        Self::Value(value)
    }
}

impl From<Const> for Expr {
    fn from(value: Const) -> Self {
        Self::Value(Value::Const(value))
    }
}

impl From<VarId> for Expr {
    fn from(var: VarId) -> Self {
        Self::Value(Value::Var(var))
    }
}

#[derive(Debug, Clone)]
pub struct Add {
    pub lhs: Value,
    pub rhs: Value,
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
    pub value: Value,
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
pub struct Neg {
    pub value: Value,
}

impl Neg {
    pub fn new<V>(value: V) -> Self
    where
        V: Into<Value>,
    {
        Self {
            value: value.into(),
        }
    }
}

impl Pretty for Neg {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text("neg")
            .append(allocator.space())
            .append(self.value.pretty(allocator))
    }
}

#[derive(Debug, Clone)]
pub struct Eq {
    pub lhs: Value,
    pub rhs: Value,
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
    pub ptr: Value,
    pub effect: EffectId,
    pub prev_effect: Option<EffectId>,
}

impl Load {
    pub fn new<E>(ptr: Value, effect: EffectId, prev_effect: E) -> Self
    where
        E: Into<Option<EffectId>>,
    {
        Self {
            ptr,
            effect,
            prev_effect: prev_effect.into(),
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
            .append(allocator.text(if let Some(prev_effect) = self.prev_effect {
                format!("// eff: {}, pred: {}", self.effect, prev_effect)
            } else {
                format!("// eff: {}, pred: ???", self.effect)
            }))
    }
}

#[derive(Debug, Clone)]
pub struct Store {
    pub ptr: Value,
    pub value: Value,
    pub effect: EffectId,
    pub prev_effect: Option<EffectId>,
}

impl Store {
    pub fn new<E>(ptr: Value, value: Value, effect: EffectId, prev_effect: E) -> Self
    where
        E: Into<Option<EffectId>>,
    {
        Self {
            ptr,
            value,
            effect,
            prev_effect: prev_effect.into(),
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
            .append(allocator.text(if let Some(prev_effect) = self.prev_effect {
                format!("// eff: {}, pred: {}", self.effect, prev_effect)
            } else {
                format!("// eff: {}, pred: ???", self.effect)
            }))
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    Var(VarId),
    Const(Const),
    Missing,
}

impl Value {
    pub fn as_var(&self) -> Option<VarId> {
        if let Self::Var(var) = *self {
            Some(var)
        } else {
            None
        }
    }

    /// Returns `true` if the value is [`Missing`].
    ///
    /// [`Missing`]: Value::Missing
    pub const fn is_missing(&self) -> bool {
        matches!(self, Self::Missing)
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
            Self::Missing => allocator.text("???"),
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
    Byte(u8),
    Bool(bool),
}

impl Const {
    pub fn equal_values(&self, other: &Self) -> bool {
        self.convert_to_i32().unwrap() == other.convert_to_i32().unwrap()
    }

    pub fn convert_to_i32(&self) -> Option<i32> {
        match *self {
            Self::Int(int) => Some(int),
            Self::Byte(byte) => Some(byte as i32),
            Self::Bool(bool) => Some(bool as i32),
        }
    }

    pub fn convert_to_u8(&self) -> Option<u8> {
        match *self {
            Self::Int(int) => Some(int.rem_euclid(u8::MAX as i32) as u8),
            Self::Byte(byte) => Some(byte),
            Self::Bool(bool) => Some(bool as u8),
        }
    }

    pub fn convert_to_u16(&self) -> Option<u16> {
        match *self {
            Self::Int(int) => Some(int.rem_euclid(u16::MAX as i32) as u16),
            Self::Byte(byte) => Some(byte as u16),
            Self::Bool(bool) => Some(bool as u16),
        }
    }

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

impl ops::Not for Const {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Self::Int(int) => Self::Int(!int),
            Self::Byte(byte) => Self::Byte(!byte),
            Self::Bool(bool) => Self::Bool(!bool),
        }
    }
}

impl ops::Not for &Const {
    type Output = Const;

    fn not(self) -> Self::Output {
        !self.clone()
    }
}

impl ops::Neg for Const {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Self::Int(int) => Self::Int(-int),
            Self::Byte(byte) => Self::Int(-(byte as i32)),
            Self::Bool(_) => panic!("cannot negate bool"),
        }
    }
}

impl ops::Neg for &Const {
    type Output = Const;

    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

impl ops::Add for Const {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Int(lhs), Self::Int(rhs)) => Self::Int(lhs + rhs),
            (Self::Int(lhs), Self::Byte(rhs)) => Self::Int(lhs + rhs as i32),
            (Self::Byte(lhs), Self::Int(rhs)) => Self::Int(lhs as i32 + rhs),
            (Self::Byte(lhs), Self::Byte(rhs)) => Self::Byte(lhs.wrapping_add(rhs)),
            (Self::Bool(_), _) | (_, Self::Bool(_)) => panic!("can't add booleans"),
        }
    }
}

impl ops::Add for &Const {
    type Output = Const;

    fn add(self, rhs: Self) -> Self::Output {
        self.clone() + rhs.clone()
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
            Self::Byte(byte) => format!("byte {}", byte),
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

impl Display for Const {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Int(int) => write!(f, "int {}", int),
            Self::Byte(byte) => write!(f, "byte {}", byte),
            Self::Bool(boolean) => write!(f, "bool {}", boolean),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct VarId(u32);

impl VarId {
    pub fn new(port: OutputPort) -> Self {
        Self(port.raw())
    }
}

impl Pretty for VarId {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator.text(format!("{}", self))
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
        f.write_char('v')?;
        Display::fmt(&self.0, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct EffectId(u32);

impl EffectId {
    pub fn new(port: OutputPort) -> Self {
        Self(port.raw())
    }
}

impl Pretty for EffectId {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator.text(format!("{}", self))
    }
}

impl Debug for EffectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("EffectId(")?;
        Debug::fmt(&self.0, f)?;
        f.write_char(')')
    }
}

impl Display for EffectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_char('e')?;
        Display::fmt(&self.0, f)
    }
}
