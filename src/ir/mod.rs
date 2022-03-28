mod builder;
mod lifetime;
mod parse;
mod pretty_print;

pub use builder::IrBuilder;
pub use pretty_print::{pretty_utils, Pretty, PrettyConfig};

use crate::{
    graph::{NodeId, OutputPort, Port},
    utils::percent_total,
    values::{Cell, Ptr},
};
use pretty::{DocAllocator, DocBuilder};
use pretty_print::COMMENT_ALIGNMENT_OFFSET;
use std::{
    collections::BTreeMap,
    fmt::{self, Debug, Display, Write},
    ops::{self, Deref, DerefMut},
    slice, vec,
};

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

    pub fn single<I>(inst: I) -> Self
    where
        I: Into<Instruction>,
    {
        Self {
            instructions: vec![inst.into()],
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

impl IntoIterator for Block {
    type Item = Instruction;
    type IntoIter = vec::IntoIter<Instruction>;

    fn into_iter(self) -> Self::IntoIter {
        self.instructions.into_iter()
    }
}

impl<'a> IntoIterator for &'a Block {
    type Item = &'a Instruction;
    type IntoIter = slice::Iter<'a, Instruction>;

    fn into_iter(self) -> Self::IntoIter {
        self.instructions.as_slice().iter()
    }
}

impl<'a> IntoIterator for &'a mut Block {
    type Item = &'a mut Instruction;
    type IntoIter = slice::IterMut<'a, Instruction>;

    fn into_iter(self) -> Self::IntoIter {
        self.instructions.as_mut_slice().iter_mut()
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
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        pretty_utils::body_block(allocator, config, false, self)
    }
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Call(Call),
    Assign(Assign),
    Theta(Theta),
    Gamma(Gamma),
    Store(Store),
    LifetimeEnd(LifetimeEnd),
}

impl Instruction {
    /// Returns `true` if the instruction is a [`LifetimeEnd`].
    ///
    /// [`LifetimeEnd`]: Instruction::LifetimeEnd
    pub const fn is_lifetime_end(&self) -> bool {
        matches!(self, Self::LifetimeEnd(..))
    }

    pub const fn as_assign(&self) -> Option<&Assign> {
        if let Self::Assign(assign) = self {
            Some(assign)
        } else {
            None
        }
    }

    pub fn is_output_param(&self) -> bool {
        self.as_assign()
            .map(Assign::is_output_param)
            .unwrap_or_default()
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
            Self::Call(call) => allocator.column(move |start_column| {
                call.pretty(allocator, config)
                    .append(if config.display_effects || config.display_invocations {
                        allocator.column(move |column| {
                            let mut comment = String::new();

                            if config.display_effects {
                                if let Some(prev_effect) = call.prev_effect {
                                    write!(
                                        comment,
                                        "// eff: {}, pred: {}",
                                        call.effect, prev_effect
                                    )
                                    .unwrap();
                                } else {
                                    write!(comment, "// eff: {}, pred: ???", call.effect).unwrap();
                                }
                            }

                            if config.display_invocations && call.invocations != 0 {
                                write!(comment, ", calls: {}", call.invocations).unwrap();

                                if let Some(total_instructions) = config.total_instructions {
                                    let percentage =
                                        percent_total(total_instructions, call.invocations);
                                    write!(comment, " ({:.02}%)", percentage).unwrap();
                                }
                            }

                            allocator
                                .space()
                                .append(allocator.text(comment))
                                .indent(
                                    COMMENT_ALIGNMENT_OFFSET.saturating_sub(column - start_column),
                                )
                                .into_doc()
                        })
                    } else {
                        allocator.nil()
                    })
                    .into_doc()
            }),

            Self::Assign(assign) => assign.pretty(allocator, config),
            Self::Theta(theta) => theta.pretty(allocator, config),
            Self::Gamma(gamma) => gamma.pretty(allocator, config),
            Self::Store(store) => store.pretty(allocator, config),
            Self::LifetimeEnd(lifetime) => lifetime.pretty(allocator, config),
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

impl From<LifetimeEnd> for Instruction {
    fn from(lifetime: LifetimeEnd) -> Self {
        Self::LifetimeEnd(lifetime)
    }
}

#[derive(Debug, Clone)]
pub struct LifetimeEnd {
    pub var: VarId,
}

impl LifetimeEnd {
    pub fn new(var: VarId) -> Self {
        Self { var }
    }
}

impl Pretty for LifetimeEnd {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text("lifetime_end")
            .append(allocator.space())
            .append(self.var.pretty(allocator, config))
    }
}

#[derive(Clone)]
pub struct Theta {
    pub node: NodeId,
    pub body: Vec<Instruction>,
    pub cond: Option<Value>,
    pub output_effect: Option<EffectId>,
    pub input_effect: Option<EffectId>,
    pub inputs: BTreeMap<VarId, Value>,
    pub outputs: BTreeMap<VarId, Value>,
    pub output_feedback: BTreeMap<VarId, VarId>,
    pub loops: usize,
    pub body_inst_count: usize,
}

impl Theta {
    #[allow(clippy::too_many_arguments)]
    pub fn new<C, E1, E2>(
        node: NodeId,
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
            node,
            body,
            cond: cond.into(),
            output_effect: output_effect.into(),
            input_effect: input_effect.into(),
            inputs,
            outputs,
            output_feedback,
            loops: 0,
            body_inst_count: 0,
        }
    }
}

impl Pretty for Theta {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        let mut comment = String::new();
        if config.display_effects {
            match (self.input_effect, self.output_effect) {
                (None, None) => {
                    write!(comment, "// node: {}, eff: ???, pred: ???", self.node)
                }
                (None, Some(output_effect)) => {
                    write!(
                        comment,
                        "// node: {}, eff: {}, pred: ???",
                        self.node, output_effect,
                    )
                }
                (Some(input_effect), None) => {
                    write!(
                        comment,
                        "// node: {}, eff: ???, pred: {}",
                        self.node, input_effect,
                    )
                }
                (Some(input_effect), Some(output_effect)) => {
                    write!(
                        comment,
                        "// node: {}, eff: {}, pred: {}",
                        self.node, output_effect, input_effect,
                    )
                }
            }
            .unwrap();
        }

        if config.display_invocations {
            if self.loops != 0 {
                write!(comment, ", loops: {}", self.loops).unwrap();
            }

            if self.body_inst_count != 0 {
                write!(comment, ", body instructions: {}", self.body_inst_count,).unwrap();

                if let Some(total_instructions) = config.total_instructions {
                    let percentage = percent_total(total_instructions, self.body_inst_count);
                    write!(comment, " ({:.02}%)", percentage).unwrap();
                }
            }
        }

        if comment.is_empty() {
            allocator.nil()
        } else {
            allocator.text(comment).append(allocator.hardline())
        }
        .append(allocator.text("do"))
        .append(allocator.space())
        .append(allocator.text("{"))
        .append(pretty_utils::body_block(
            allocator, config, true, &self.body,
        ))
        .append(allocator.text("}"))
        .append(allocator.space())
        .append(allocator.text("while"))
        .append(allocator.space())
        .append(allocator.text("{"))
        .append(allocator.space())
        .append(if let Some(cond) = self.cond.as_ref() {
            cond.pretty(allocator, config)
        } else {
            allocator.text("???")
        })
        .append(allocator.space())
        .append(allocator.text("}"))
    }
}

impl Debug for Theta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let alternative = f.alternate();

        f.debug_struct("Theta")
            .field("node", &self.node)
            .field("body", if alternative { &self.body } else { &Omitted })
            .field("cond", &self.cond)
            .field("output_effect", &self.output_effect)
            .field("input_effect", &self.input_effect)
            // TODO: Could probably do better formatting for inputs & outputs
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("output_feedback", &self.output_feedback)
            .finish_non_exhaustive()
    }
}

// FIXME: Inputs
#[derive(Clone)]
pub struct Gamma {
    pub node: NodeId,
    pub cond: Value,
    pub true_branch: Vec<Instruction>,
    pub true_outputs: BTreeMap<VarId, Value>,
    pub false_branch: Vec<Instruction>,
    pub false_outputs: BTreeMap<VarId, Value>,
    pub effect: EffectId,
    pub prev_effect: Option<EffectId>,
    pub true_branches: usize,
    pub false_branches: usize,
}

impl Gamma {
    #[allow(clippy::too_many_arguments)]
    pub fn new<C, E>(
        node: NodeId,
        cond: C,
        true_branch: Vec<Instruction>,
        true_outputs: BTreeMap<VarId, Value>,
        false_branch: Vec<Instruction>,
        false_outputs: BTreeMap<VarId, Value>,
        effect: EffectId,
        prev_effect: E,
    ) -> Self
    where
        C: Into<Value>,
        E: Into<Option<EffectId>>,
    {
        Self {
            node,
            cond: cond.into(),
            true_branch,
            true_outputs,
            false_branch,
            false_outputs,
            effect,
            prev_effect: prev_effect.into(),
            true_branches: 0,
            false_branches: 0,
        }
    }

    pub fn true_is_empty(&self) -> bool {
        self.true_branch.iter().all(|inst| {
            matches!(
                inst,
                Instruction::Assign(Assign {
                    tag: AssignTag::InputParam(_) | AssignTag::OutputParam,
                    ..
                }) | Instruction::LifetimeEnd(_),
            )
        })
    }

    pub fn false_is_empty(&self) -> bool {
        self.false_branch.iter().all(|inst| {
            matches!(
                inst,
                Instruction::Assign(Assign {
                    tag: AssignTag::InputParam(_) | AssignTag::OutputParam,
                    ..
                }) | Instruction::LifetimeEnd(_),
            )
        })
    }
}

impl Pretty for Gamma {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        let mut comment = String::new();

        if config.display_effects {
            if let Some(prev_effect) = self.prev_effect {
                write!(
                    comment,
                    "// node: {}, eff: {}, pred: {}",
                    self.node, self.effect, prev_effect,
                )
            } else {
                write!(
                    comment,
                    "// node: {}, eff: {}, pred: ???",
                    self.node, self.effect,
                )
            }
            .unwrap();
        }

        if config.display_invocations {
            if self.true_branches + self.false_branches != 0 {
                if config.display_effects {
                    comment.push_str(" ,");
                }

                write!(
                    comment,
                    "branches: {}",
                    self.true_branches + self.false_branches,
                )
                .unwrap();
            }

            if self.true_branches != 0 {
                write!(comment, ", true branches: {}", self.true_branches).unwrap();
            }

            if self.false_branches != 0 {
                write!(comment, ", false branches: {}", self.false_branches).unwrap();
            }
        }

        if comment.is_empty() {
            allocator.nil()
        } else {
            allocator.text(comment).append(allocator.hardline())
        }
        .append(allocator.text("if"))
        .append(allocator.space())
        .append(self.cond.pretty(allocator, config))
        .append(allocator.space())
        .append(allocator.text("{"))
        .append(pretty_utils::body_block(
            allocator,
            config,
            true,
            &self.true_branch,
        ))
        .append(allocator.text("}"))
        .append(allocator.space())
        .append(allocator.text("else"))
        .append(allocator.space())
        .append(allocator.text("{"))
        .append(pretty_utils::body_block(
            allocator,
            config,
            true,
            &self.false_branch,
        ))
        .append(allocator.text("}"))
    }
}

impl Debug for Gamma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let alternative = f.alternate();

        f.debug_struct("Gamma")
            .field("node", &self.node)
            .field("cond", &self.cond)
            .field(
                "true_branch",
                if alternative {
                    &self.true_branch
                } else {
                    &Omitted
                },
            )
            // TODO: Could probably do better formatting for inputs & outputs
            .field("true_outputs", &self.true_outputs)
            .field(
                "false_branch",
                if alternative {
                    &self.false_branch
                } else {
                    &Omitted
                },
            )
            // TODO: Could probably do better formatting for inputs & outputs
            .field("false_outputs", &self.false_outputs)
            .field("effect", &self.effect)
            .field("prev_effect", &self.prev_effect)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub struct Call {
    pub node: NodeId,
    pub function: CallFunction,
    pub args: Vec<Value>,
    pub effect: EffectId,
    pub prev_effect: Option<EffectId>,
    pub invocations: usize,
}

impl Call {
    pub fn new<E>(
        node: NodeId,
        function: CallFunction,
        args: Vec<Value>,
        effect: EffectId,
        prev_effect: E,
    ) -> Self
    where
        E: Into<Option<EffectId>>,
    {
        Self {
            node,
            function,
            args,
            effect,
            prev_effect: prev_effect.into(),
            invocations: 0,
        }
    }
}

impl Pretty for Call {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator.text("call").append(allocator.space()).append(
            allocator.text(self.function.to_str()).append(
                allocator
                    .intersperse(
                        self.args.iter().map(|arg| arg.pretty(allocator, config)),
                        allocator.text(",").append(allocator.space()),
                    )
                    .parens(),
            ),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallFunction {
    Input,
    Output,
    Scanr,
    Scanl,
}

impl CallFunction {
    pub const fn to_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
            Self::Scanr => "scanr",
            Self::Scanl => "scanl",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Assign {
    pub var: VarId,
    pub value: Expr,
    pub tag: AssignTag,
    pub invocations: usize,
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
            invocations: 0,
        }
    }

    pub fn input<I>(var: VarId, value: I, variance: Variance) -> Self
    where
        I: Into<Expr>,
    {
        Self {
            var,
            value: value.into(),
            tag: AssignTag::InputParam(variance),
            invocations: 0,
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
            invocations: 0,
        }
    }

    pub fn is_input_param(&self) -> bool {
        self.tag.is_input_param()
    }

    pub fn is_output_param(&self) -> bool {
        self.tag.is_output_param()
    }
}

impl Pretty for Assign {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator.column(move |start_column| {
            self.var
                .pretty(allocator, config)
                .append(allocator.space())
                .append(allocator.text(":="))
                .append(allocator.space())
                .append(match self.tag {
                    AssignTag::None => allocator.nil(),
                    AssignTag::InputParam(_) => allocator.text("in").append(allocator.space()),
                    AssignTag::OutputParam => allocator.text("out").append(allocator.space()),
                })
                .append(self.value.pretty(
                    allocator,
                    if self.value.is_call() {
                        PrettyConfig::minimal()
                    } else {
                        config
                    },
                ))
                .append(if let Expr::Load(load) = &self.value {
                    if config.display_effects || config.display_invocations {
                        allocator.column(move |column| {
                            let mut comment = String::new();

                            if config.display_effects {
                                if let Some(prev_effect) = load.prev_effect {
                                    write!(
                                        comment,
                                        "// eff: {}, pred: {}",
                                        load.effect, prev_effect,
                                    )
                                } else {
                                    write!(comment, "// eff: {}, pred: ???", load.effect)
                                }
                                .unwrap();
                            }

                            if config.display_invocations && self.invocations != 0 {
                                write!(comment, ", loads: {}", self.invocations).unwrap();

                                if let Some(total_instructions) = config.total_instructions {
                                    let percentage =
                                        percent_total(total_instructions, self.invocations);
                                    write!(comment, " ({:.02}%)", percentage).unwrap();
                                }
                            }

                            allocator
                                .space()
                                .append(allocator.text(comment).indent(
                                    COMMENT_ALIGNMENT_OFFSET.saturating_sub(column - start_column),
                                ))
                                .into_doc()
                        })
                    } else {
                        allocator.nil()
                    }
                } else if let Expr::Call(call) = &self.value {
                    if config.display_effects || config.display_invocations {
                        allocator.column(move |column| {
                            let mut comment = if let Some(prev_effect) = call.prev_effect {
                                format!("// eff: {}, pred: {}", call.effect, prev_effect)
                            } else {
                                format!("// eff: {}, pred: ???", call.effect)
                            };

                            if call.invocations != 0 {
                                write!(comment, ", calls: {}", call.invocations).unwrap();

                                if let Some(total_instructions) = config.total_instructions {
                                    let percentage =
                                        percent_total(total_instructions, call.invocations);
                                    write!(comment, " ({:.02}%)", percentage).unwrap();
                                }
                            }

                            allocator
                                .space()
                                .append(allocator.text(comment))
                                .indent(
                                    COMMENT_ALIGNMENT_OFFSET.saturating_sub(column - start_column),
                                )
                                .into_doc()
                        })
                    } else {
                        allocator.nil()
                    }
                } else {
                    allocator.nil()
                })
                .append(allocator.column(move |column| {
                    if let AssignTag::InputParam(variance) = self.tag {
                        if config.display_invocations {
                            match variance {
                                Variance::Invariant => {
                                    allocator.space().append(allocator.text("// invariant"))
                                }

                                Variance::Variant { feedback_from } => allocator.space().append(
                                    allocator
                                        .text(format!("// variant, feedback: {}", feedback_from)),
                                ),

                                Variance::None => {
                                    if self.invocations != 0 && !self.value.is_call() {
                                        let mut comment =
                                            format!("// invocations: {}", self.invocations);

                                        if let Some(total_instructions) = config.total_instructions
                                        {
                                            let percentage =
                                                percent_total(total_instructions, self.invocations);
                                            write!(comment, " ({:.02}%)", percentage).unwrap();
                                        }

                                        allocator.space().append(allocator.text(comment))
                                    } else {
                                        return allocator.nil().into_doc();
                                    }
                                }
                            }
                            .indent(COMMENT_ALIGNMENT_OFFSET.saturating_sub(column - start_column))
                        } else {
                            return allocator.nil().into_doc();
                        }
                    } else if self.tag == AssignTag::None
                        && !self.value.is_load()
                        && !self.value.is_call()
                        && config.display_invocations
                    {
                        if self.invocations != 0 && config.display_invocations {
                            let mut comment = format!("// invocations: {}", self.invocations);

                            if let Some(total_instructions) = config.total_instructions {
                                let percentage =
                                    percent_total(total_instructions, self.invocations);
                                write!(comment, " ({:.02}%)", percentage).unwrap();
                            }

                            allocator.space().append(allocator.text(comment))
                        } else {
                            return allocator.nil().into_doc();
                        }
                        .indent(COMMENT_ALIGNMENT_OFFSET.saturating_sub(column - start_column))
                    } else {
                        allocator.nil()
                    }
                    .into_doc()
                }))
                .into_doc()
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssignTag {
    None,
    InputParam(Variance),
    OutputParam,
}

impl AssignTag {
    /// Returns `true` if the assign tag is [`InputParam`].
    ///
    /// [`InputParam`]: AssignTag::InputParam
    pub const fn is_input_param(&self) -> bool {
        matches!(self, Self::InputParam(..))
    }

    /// Returns `true` if the assign tag is [`OutputParam`].
    ///
    /// [`OutputParam`]: AssignTag::OutputParam
    pub const fn is_output_param(&self) -> bool {
        matches!(self, Self::OutputParam)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Variance {
    Invariant,
    Variant { feedback_from: VarId },
    None,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Cmp(Cmp),
    Add(Add),
    Sub(Sub),
    Mul(Mul),
    Not(Not),
    Neg(Neg),
    Load(Load),
    Call(Call),
    Value(Value),
}

impl Expr {
    /// Returns `true` if the expr is a [`Load`].
    ///
    /// [`Load`]: Expr::Load
    pub const fn is_load(&self) -> bool {
        matches!(self, Self::Load(..))
    }

    /// Returns `true` if the expr is [`Call`].
    ///
    /// [`Call`]: Expr::Call
    pub const fn is_call(&self) -> bool {
        matches!(self, Self::Call(..))
    }

    pub const fn is_const(&self) -> bool {
        matches!(self, Self::Value(Value::Const(_)))
    }
}

impl Pretty for Expr {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        match self {
            Self::Cmp(cmp) => cmp.pretty(allocator, config),
            Self::Add(add) => add.pretty(allocator, config),
            Self::Sub(sub) => sub.pretty(allocator, config),
            Self::Mul(mul) => mul.pretty(allocator, config),
            Self::Not(not) => not.pretty(allocator, config),
            Self::Neg(neg) => neg.pretty(allocator, config),
            Self::Load(load) => load.pretty(allocator, config),
            Self::Call(call) => call.pretty(allocator, config),
            Self::Value(value) => value.pretty(allocator, config),
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

impl From<Cmp> for Expr {
    fn from(cmp: Cmp) -> Self {
        Self::Cmp(cmp)
    }
}

impl From<Add> for Expr {
    fn from(add: Add) -> Self {
        Self::Add(add)
    }
}

impl From<Sub> for Expr {
    fn from(sub: Sub) -> Self {
        Self::Sub(sub)
    }
}

impl From<Mul> for Expr {
    fn from(mul: Mul) -> Self {
        Self::Mul(mul)
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
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text("add")
            .append(allocator.space())
            .append(self.lhs.pretty(allocator, config))
            .append(allocator.text(","))
            .append(allocator.space())
            .append(self.rhs.pretty(allocator, config))
    }
}

#[derive(Debug, Clone)]
pub struct Sub {
    pub lhs: Value,
    pub rhs: Value,
}

impl Sub {
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

impl Pretty for Sub {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text("sub")
            .append(allocator.space())
            .append(self.lhs.pretty(allocator, config))
            .append(allocator.text(","))
            .append(allocator.space())
            .append(self.rhs.pretty(allocator, config))
    }
}

#[derive(Debug, Clone)]
pub struct Mul {
    pub lhs: Value,
    pub rhs: Value,
}

impl Mul {
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

impl Pretty for Mul {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text("mul")
            .append(allocator.space())
            .append(self.lhs.pretty(allocator, config))
            .append(allocator.text(","))
            .append(allocator.space())
            .append(self.rhs.pretty(allocator, config))
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
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text("not")
            .append(allocator.space())
            .append(self.value.pretty(allocator, config))
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
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text("neg")
            .append(allocator.space())
            .append(self.value.pretty(allocator, config))
    }
}

#[derive(Debug, Clone)]
pub struct Cmp {
    pub lhs: Value,
    pub rhs: Value,
    pub op: CmpKind,
}

impl Cmp {
    pub fn new<L, R>(lhs: L, rhs: R, op: CmpKind) -> Self
    where
        L: Into<Value>,
        R: Into<Value>,
    {
        Self {
            lhs: lhs.into(),
            rhs: rhs.into(),
            op,
        }
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
            .append(self.op.pretty(allocator, config))
            .append(allocator.space())
            .append(self.lhs.pretty(allocator, config))
            .append(allocator.text(","))
            .append(allocator.space())
            .append(self.rhs.pretty(allocator, config))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpKind {
    Eq,
    Neq,
    Less,
    Greater,
    LessEq,
    GreaterEq,
}

impl CmpKind {
    pub const fn operator(&self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::Neq => "!=",
            Self::Less => "<",
            Self::Greater => ">",
            Self::LessEq => "<=",
            Self::GreaterEq => ">=",
        }
    }
}

impl Pretty for CmpKind {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, _config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        match self {
            Self::Eq => allocator.text("eq"),
            Self::Neq => allocator.text("neq"),
            Self::Less => allocator.text("lt"),
            Self::Greater => allocator.text("gt"),
            Self::LessEq => allocator.text("le"),
            Self::GreaterEq => allocator.text("ge"),
        }
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
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator
            .text("load")
            .append(allocator.space())
            .append(self.ptr.pretty(allocator, config))
    }
}

#[derive(Debug, Clone)]
pub struct Store {
    pub ptr: Value,
    pub value: Value,
    pub effect: EffectId,
    pub prev_effect: Option<EffectId>,
    pub stores: usize,
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
            stores: 0,
        }
    }
}

impl Pretty for Store {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        allocator.column(move |start_column| {
            allocator
                .text("store")
                .append(allocator.space())
                .append(self.ptr.pretty(allocator, config))
                .append(allocator.text(","))
                .append(allocator.space())
                .append(self.value.pretty(allocator, config))
                .append(if config.display_effects || config.display_invocations {
                    allocator.column(move |column| {
                        let mut comment = if let Some(prev_effect) = self.prev_effect {
                            format!("// eff: {}, pred: {}", self.effect, prev_effect)
                        } else {
                            format!("// eff: {}, pred: ???", self.effect)
                        };

                        if self.stores != 0 {
                            write!(comment, ", stores: {}", self.stores).unwrap();

                            if let Some(total_instructions) = config.total_instructions {
                                let percentage = percent_total(total_instructions, self.stores);
                                write!(comment, " ({:.02}%)", percentage).unwrap();
                            }
                        }

                        allocator
                            .space()
                            .append(allocator.text(comment))
                            .indent(COMMENT_ALIGNMENT_OFFSET.saturating_sub(column - start_column))
                            .into_doc()
                    })
                } else {
                    allocator.nil()
                })
                .into_doc()
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Var(VarId),
    Const(Const),
    Missing,
}

impl Value {
    /// Returns `true` if the value is [`Missing`].
    ///
    /// [`Missing`]: Value::Missing
    pub const fn is_missing(&self) -> bool {
        matches!(self, Self::Missing)
    }

    pub const fn as_var(&self) -> Option<VarId> {
        if let Self::Var(var) = *self {
            Some(var)
        } else {
            None
        }
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
            Self::Var(var) => var.pretty(allocator, config),
            Self::Const(constant) => constant.pretty(allocator, config),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Const {
    Ptr(Ptr),
    Cell(Cell),
    Bool(bool),
}

impl Const {
    pub fn into_ptr(self, tape_len: u16) -> Ptr {
        match self {
            Self::Ptr(ptr) => ptr,
            Self::Cell(cell) => cell.into_ptr(tape_len),
            Self::Bool(_) => unreachable!(),
        }
    }

    pub fn into_cell(self) -> Cell {
        match self {
            Self::Ptr(ptr) => ptr.into_cell(),
            Self::Cell(cell) => cell,
            Self::Bool(_) => unreachable!(),
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(bool) = *self {
            Some(bool)
        } else {
            None
        }
    }

    pub fn as_ptr(&self) -> Option<Ptr> {
        if let Self::Ptr(ptr) = *self {
            Some(ptr)
        } else {
            None
        }
    }
}

impl ops::Not for Const {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Self::Ptr(ptr) => Self::Ptr(!ptr),
            Self::Cell(cell) => Self::Cell(!cell),
            Self::Bool(bool) => Self::Bool(!bool),
        }
    }
}

impl ops::Not for &Const {
    type Output = Const;

    fn not(self) -> Self::Output {
        !*self
    }
}

impl ops::Neg for Const {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Self::Ptr(int) => Self::Ptr(Ptr::zero(int.tape_len()) - int),
            Self::Cell(byte) => Self::Cell(byte.wrapping_neg()),
            Self::Bool(_) => panic!("cannot negate bool"),
        }
    }
}

impl ops::Neg for &Const {
    type Output = Const;

    fn neg(self) -> Self::Output {
        -*self
    }
}

impl ops::Add for Const {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Ptr(lhs), Self::Ptr(rhs)) => Self::Ptr(lhs + rhs),
            (Self::Ptr(lhs), Self::Cell(rhs)) => Self::Ptr(lhs + rhs),
            (Self::Cell(lhs), Self::Ptr(rhs)) => Self::Ptr(lhs + rhs),
            (Self::Cell(lhs), Self::Cell(rhs)) => Self::Cell(lhs + rhs),
            (Self::Bool(_), _) | (_, Self::Bool(_)) => panic!("can't add booleans"),
        }
    }
}

impl ops::Add for &Const {
    type Output = Const;

    fn add(self, rhs: Self) -> Self::Output {
        *self + *rhs
    }
}

impl ops::Sub for Const {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Ptr(lhs), Self::Ptr(rhs)) => Self::Ptr(lhs - rhs),
            (Self::Ptr(lhs), Self::Cell(rhs)) => Self::Ptr(lhs - rhs),
            (Self::Cell(lhs), Self::Ptr(rhs)) => Self::Ptr(lhs - rhs),
            (Self::Cell(lhs), Self::Cell(rhs)) => Self::Cell(lhs - rhs),
            (Self::Bool(_), _) | (_, Self::Bool(_)) => panic!("can't subtract booleans"),
        }
    }
}

impl ops::Sub for &Const {
    type Output = Const;

    fn sub(self, rhs: Self) -> Self::Output {
        *self - *rhs
    }
}

impl ops::Mul for Const {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Ptr(lhs), Self::Ptr(rhs)) => Self::Ptr(lhs * rhs),
            (Self::Ptr(lhs), Self::Cell(rhs)) => Self::Ptr(lhs * rhs),
            (Self::Cell(lhs), Self::Ptr(rhs)) => Self::Ptr(lhs * rhs),
            (Self::Cell(lhs), Self::Cell(rhs)) => Self::Cell(lhs * rhs),
            (Self::Bool(_), _) | (_, Self::Bool(_)) => panic!("can't multiply booleans"),
        }
    }
}

impl ops::Mul for &Const {
    type Output = Const;

    fn mul(self, rhs: Self) -> Self::Output {
        *self * *rhs
    }
}

impl Pretty for Const {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, _config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        let text = match *self {
            Self::Ptr(int) => format!("int {}", int),
            Self::Cell(byte) => format!("byte {}", byte),
            Self::Bool(boolean) => format!("bool {}", boolean),
        };
        allocator.text(text)
    }
}

impl From<Ptr> for Const {
    fn from(ptr: Ptr) -> Self {
        Self::Ptr(ptr)
    }
}

impl From<Cell> for Const {
    fn from(cell: Cell) -> Self {
        Self::Cell(cell)
    }
}

impl From<u8> for Const {
    fn from(byte: u8) -> Self {
        Self::Cell(Cell::new(byte))
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
            Self::Ptr(int) => write!(f, "int {}", int),
            Self::Cell(byte) => write!(f, "byte {}", byte),
            Self::Bool(boolean) => write!(f, "bool {}", boolean),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct VarId(pub u32);

impl VarId {
    pub fn new(port: OutputPort) -> Self {
        Self(port.raw())
    }
}

impl Pretty for VarId {
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, _config: PrettyConfig) -> DocBuilder<'a, D, A>
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
    fn pretty<'a, D, A>(&'a self, allocator: &'a D, _config: PrettyConfig) -> DocBuilder<'a, D, A>
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

struct Omitted;

impl Debug for Omitted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("...")
    }
}
