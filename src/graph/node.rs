use crate::graph::{
    Add, Bool, EdgeCount, EdgeDescriptor, EdgeKind, End, Eq, Gamma, Input, InputParam, InputPort,
    Int, Load, Neg, NodeExt, NodeId, Not, Output, OutputParam, OutputPort, Start, Store, Theta,
};
use tinyvec::{tiny_vec, TinyVec};

// TODO: derive_more?
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Int(Int, i32),
    Bool(Bool, bool),
    Add(Add),
    Load(Load),
    Store(Store),
    Start(Start),
    End(End),
    Input(Input),
    Output(Output),
    Theta(Box<Theta>),
    InputPort(InputParam),
    OutputPort(OutputParam),
    Eq(Eq),
    Not(Not),
    Neg(Neg),
    Gamma(Box<Gamma>),
}

impl Node {
    pub fn node_id(&self) -> NodeId {
        match self {
            Self::Int(Int { node, .. }, _)
            | Self::Bool(Bool { node, .. }, _)
            | Self::Add(Add { node, .. })
            | Self::Load(Load { node, .. })
            | Self::Store(Store { node, .. })
            | Self::Start(Start { node, .. })
            | Self::End(End { node, .. })
            | Self::Input(Input { node, .. })
            | Self::Output(Output { node, .. })
            | Self::InputPort(InputParam { node, .. })
            | Self::OutputPort(OutputParam { node, .. })
            | Self::Eq(Eq { node, .. })
            | Self::Not(Not { node, .. })
            | Self::Neg(Neg { node, .. }) => *node,
            Self::Gamma(gamma) => gamma.node(),
            Self::Theta(theta) => theta.node(),
        }
    }

    // FIXME: TinyVec?
    pub fn inputs(&self) -> TinyVec<[InputPort; 4]> {
        match self {
            Self::Int(_, _) | Self::Bool(_, _) => TinyVec::new(),
            Self::Add(add) => tiny_vec![add.lhs, add.rhs],
            Self::Load(load) => tiny_vec![load.ptr, load.effect_in],
            Self::Store(store) => tiny_vec![store.ptr, store.value, store.effect_in],
            Self::Start(_) => TinyVec::new(),
            Self::End(end) => tiny_vec![end.effect],
            Self::Input(input) => tiny_vec![input.effect_in],
            Self::Output(output) => tiny_vec![output.value, output.effect_in],
            Self::Theta(theta) => theta.all_input_ports(),
            Self::InputPort(_) => TinyVec::new(),
            Self::OutputPort(output) => tiny_vec![output.input],
            Self::Eq(eq) => tiny_vec![eq.lhs, eq.rhs],
            Self::Not(not) => tiny_vec![not.input],
            Self::Neg(neg) => tiny_vec![neg.input],
            Self::Gamma(gamma) => {
                let mut inputs = TinyVec::with_capacity(gamma.inputs().len() + 2);
                inputs.extend(gamma.inputs().iter().copied());
                inputs.push(gamma.condition());
                inputs.push(gamma.effect_in());
                inputs
            }
        }
    }

    // FIXME: Remove this and use a setter
    pub fn inputs_mut(&mut self) -> Vec<&mut InputPort> {
        match self {
            Self::Int(_, _) | Self::Bool(_, _) => Vec::new(),
            Self::Add(add) => vec![&mut add.lhs, &mut add.rhs],
            Self::Load(load) => vec![&mut load.ptr, &mut load.effect_in],
            Self::Store(store) => vec![&mut store.ptr, &mut store.value, &mut store.effect_in],
            Self::Start(_) => Vec::new(),
            Self::End(end) => vec![&mut end.effect],
            Self::Input(input) => vec![&mut input.effect_in],
            Self::Output(output) => vec![&mut output.value, &mut output.effect_in],
            Self::Theta(_) => {
                tracing::warn!("use setters for setting inputs on theta nodes");
                Vec::new()
            }
            Self::InputPort(_) => Vec::new(),
            Self::OutputPort(output) => vec![&mut output.input],
            Self::Eq(eq) => vec![&mut eq.lhs, &mut eq.rhs],
            Self::Not(not) => vec![&mut not.input],
            Self::Neg(neg) => vec![&mut neg.input],
            Self::Gamma(gamma) => {
                let mut inputs: Vec<_> = gamma.inputs.iter_mut().collect();
                inputs.push(&mut gamma.condition);
                inputs.push(&mut gamma.effect_in);
                inputs
            }
        }
    }

    pub fn outputs(&self) -> TinyVec<[OutputPort; 4]> {
        match self {
            Self::Int(int, _) => tiny_vec![int.value],
            Self::Bool(bool, _) => tiny_vec![bool.value],
            Self::Add(add) => tiny_vec![add.value],
            Self::Load(load) => tiny_vec![load.value, load.effect_out],
            Self::Store(store) => tiny_vec![store.effect_out],
            Self::Start(start) => tiny_vec![start.effect],
            Self::End(_) => TinyVec::new(),
            Self::Input(input) => tiny_vec![input.value, input.effect_out],
            Self::Output(output) => tiny_vec![output.effect_out],
            Self::Theta(theta) => theta.all_output_ports(),
            Self::InputPort(input) => tiny_vec![input.output],
            Self::OutputPort(_) => TinyVec::new(),
            Self::Eq(eq) => tiny_vec![eq.value],
            Self::Not(not) => tiny_vec![not.value],
            Self::Neg(neg) => tiny_vec![neg.value],
            Self::Gamma(gamma) => {
                let mut inputs = TinyVec::with_capacity(gamma.outputs().len() + 1);
                inputs.extend(gamma.outputs().iter().copied());
                inputs.push(gamma.effect_out());
                inputs
            }
        }
    }

    // FIXME: Remove this and use a setter
    pub fn outputs_mut(&mut self) -> Vec<&mut OutputPort> {
        match self {
            Self::Int(int, _) => vec![&mut int.value],
            Self::Bool(bool, _) => vec![&mut bool.value],
            Self::Add(add) => vec![&mut add.value],
            Self::Load(load) => vec![&mut load.value, &mut load.effect_out],
            Self::Store(store) => vec![&mut store.effect_out],
            Self::Start(start) => vec![&mut start.effect],
            Self::End(_) => Vec::new(),
            Self::Input(input) => vec![&mut input.value, &mut input.effect_out],
            Self::Output(output) => vec![&mut output.effect_out],
            Self::Theta(_) => {
                tracing::warn!("use setters for setting outputs on theta nodes");
                Vec::new()
            }
            Self::InputPort(input) => vec![&mut input.output],
            Self::OutputPort(_) => Vec::new(),
            Self::Eq(eq) => vec![&mut eq.value],
            Self::Not(not) => vec![&mut not.value],
            Self::Neg(neg) => vec![&mut neg.value],
            Self::Gamma(gamma) => {
                let mut inputs: Vec<_> = gamma.outputs.iter_mut().collect();
                inputs.push(&mut gamma.effect_out);
                inputs
            }
        }
    }

    pub fn input_desc(&self) -> EdgeDescriptor {
        match self {
            Self::Int(..) | Self::Bool(..) | Self::Start(_) | Self::InputPort(_) => {
                EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::zero())
            }
            Self::Add(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::two()),
            Self::Load(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::one()),
            Self::Store(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::two()),
            Self::End(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
            Self::Input(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
            Self::Output(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::one()),
            Self::Theta(theta) => theta.input_desc(),
            Self::OutputPort(output) => match output.kind {
                EdgeKind::Effect => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
                EdgeKind::Value => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
            },
            Self::Eq(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::two()),
            Self::Not(_) | Self::Neg(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
            Self::Gamma(gamma) => {
                EdgeDescriptor::new(EdgeCount::one(), EdgeCount::exact(gamma.inputs().len() + 1))
            }
        }
    }

    pub fn output_desc(&self) -> EdgeDescriptor {
        match self {
            Self::Int(..) | Self::Bool(..) | Self::Add(_) | Self::OutputPort(_) => {
                EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::unlimited())
            }
            Self::Load(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::unlimited()),
            Self::Store(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
            Self::Start(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
            Self::End(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::zero()),
            Self::Input(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::one()),
            Self::Output(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
            Self::Theta(theta) => theta.output_desc(),
            Self::InputPort(output) => match output.kind {
                EdgeKind::Effect => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
                EdgeKind::Value => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
            },
            Self::Eq(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
            Self::Not(_) | Self::Neg(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
            Self::Gamma(gamma) => {
                EdgeDescriptor::new(EdgeCount::one(), EdgeCount::exact(gamma.outputs().len()))
            }
        }
    }

    /// Returns `true` if the node is [`Int`].
    ///
    /// [`Int`]: Node::Int
    pub const fn is_int(&self) -> bool {
        matches!(self, Self::Int(..))
    }

    /// Returns `true` if the node is [`Store`].
    ///
    /// [`Store`]: Node::Store
    pub const fn is_store(&self) -> bool {
        matches!(self, Self::Store(..))
    }

    /// Returns `true` if the node is [`End`].
    ///
    /// [`End`]: Node::End
    pub const fn is_end(&self) -> bool {
        matches!(self, Self::End(..))
    }

    /// Returns `true` if the node is [`Start`].
    ///
    /// [`Start`]: Node::Start
    pub const fn is_start(&self) -> bool {
        matches!(self, Self::Start(..))
    }

    /// Returns `true` if the node is an [`InputPort`].
    ///
    /// [`InputPort`]: Node::InputPort
    pub const fn is_input_port(&self) -> bool {
        matches!(self, Self::InputPort(..))
    }

    /// Returns `true` if the node is an [`OutputPort`].
    ///
    /// [`OutputPort`]: Node::OutputPort
    pub const fn is_output_port(&self) -> bool {
        matches!(self, Self::OutputPort(..))
    }

    pub const fn is_eq(&self) -> bool {
        matches!(self, Self::Eq(..))
    }

    pub const fn as_int(&self) -> Option<(Int, i32)> {
        if let Self::Int(int, val) = *self {
            Some((int, val))
        } else {
            None
        }
    }

    pub const fn as_bool(&self) -> Option<(Bool, bool)> {
        if let Self::Bool(bool, val) = *self {
            Some((bool, val))
        } else {
            None
        }
    }

    pub const fn as_store(&self) -> Option<Store> {
        if let Self::Store(store) = *self {
            Some(store)
        } else {
            None
        }
    }

    pub const fn as_load(&self) -> Option<Load> {
        if let Self::Load(load) = *self {
            Some(load)
        } else {
            None
        }
    }

    pub const fn as_end(&self) -> Option<End> {
        if let Self::End(end) = *self {
            Some(end)
        } else {
            None
        }
    }

    pub const fn as_add(&self) -> Option<Add> {
        if let Self::Add(add) = *self {
            Some(add)
        } else {
            None
        }
    }

    pub const fn as_eq(&self) -> Option<Eq> {
        if let Self::Eq(eq) = *self {
            Some(eq)
        } else {
            None
        }
    }

    pub const fn as_not(&self) -> Option<Not> {
        if let Self::Not(not) = *self {
            Some(not)
        } else {
            None
        }
    }

    pub const fn as_theta(&self) -> Option<&Theta> {
        if let Self::Theta(theta) = self {
            Some(theta)
        } else {
            None
        }
    }

    pub const fn as_input_param(&self) -> Option<InputParam> {
        if let Self::InputPort(param) = *self {
            Some(param)
        } else {
            None
        }
    }

    pub const fn as_output_param(&self) -> Option<OutputParam> {
        if let Self::OutputPort(param) = *self {
            Some(param)
        } else {
            None
        }
    }

    #[track_caller]
    pub fn to_start(&self) -> Start {
        if let Self::Start(start) = *self {
            start
        } else {
            panic!("attempted to get start, got {:?}", self);
        }
    }

    #[track_caller]
    pub fn to_add(&self) -> Add {
        if let Self::Add(add) = *self {
            add
        } else {
            panic!("attempted to get add, got {:?}", self);
        }
    }

    #[track_caller]
    #[allow(dead_code)]
    pub fn to_add_mut(&mut self) -> &mut Add {
        if let Self::Add(add) = self {
            add
        } else {
            panic!("attempted to get add, got {:?}", self);
        }
    }

    #[track_caller]
    #[allow(dead_code)]
    pub fn to_gamma_mut(&mut self) -> &mut Gamma {
        if let Self::Gamma(gamma) = self {
            gamma
        } else {
            panic!("attempted to get gamma, got {:?}", self);
        }
    }

    #[track_caller]
    #[allow(dead_code)]
    pub fn to_store_mut(&mut self) -> &mut Store {
        if let Self::Store(store) = self {
            store
        } else {
            panic!("attempted to get store, got {:?}", self);
        }
    }

    #[track_caller]
    #[allow(dead_code)]
    pub fn to_int(&self) -> Int {
        if let Self::Int(int, _) = *self {
            int
        } else {
            panic!("attempted to get int, got {:?}", self);
        }
    }

    #[track_caller]
    #[allow(dead_code)]
    pub fn to_int_value(&self) -> i32 {
        if let Self::Int(_, int) = *self {
            int
        } else {
            panic!("attempted to get int, got {:?}", self);
        }
    }

    #[track_caller]
    #[allow(dead_code)]
    pub fn to_bool(&self) -> Bool {
        if let Self::Bool(bool, _) = *self {
            bool
        } else {
            panic!("attempted to get bool, got {:?}", self);
        }
    }

    #[track_caller]
    #[allow(dead_code)]
    pub fn to_bool_val(&self) -> bool {
        if let Self::Bool(_, bool) = *self {
            bool
        } else {
            panic!("attempted to get bool, got {:?}", self);
        }
    }

    #[track_caller]
    pub fn to_input_param(&self) -> InputParam {
        if let Self::InputPort(param) = *self {
            param
        } else {
            panic!("attempted to get input port, got {:?}", self);
        }
    }

    #[track_caller]
    pub fn to_output_param(&self) -> OutputParam {
        if let Self::OutputPort(param) = *self {
            param
        } else {
            panic!("attempted to get output port, got {:?}", self);
        }
    }

    #[track_caller]
    #[allow(dead_code)]
    pub fn to_theta_mut(&mut self) -> &mut Theta {
        if let Self::Theta(theta) = self {
            theta
        } else {
            panic!("attempted to get theta, got {:?}", self);
        }
    }
}

macro_rules! node_traits {
    ($($variant:ident),* $(,)?) => {
        $(
            impl From<$variant> for Node {
                fn from(node: $variant) -> Self {
                    Self::$variant(node)
                }
            }

            impl TryInto<$variant> for Node {
                type Error = Self;

                fn try_into(self) -> Result<$variant, Self::Error> {
                    if let Self::$variant(node) = self {
                        Ok(node)
                    } else {
                        Err(self)
                    }
                }
            }

            impl TryInto<$variant> for &Node {
                type Error = Self;

                fn try_into(self) -> Result<$variant, Self::Error> {
                    if let Node::$variant(node) = *self {
                        Ok(node)
                    } else {
                        Err(self)
                    }
                }
            }
        )*
    };
}

node_traits! {
    Add,
    Load,
    Store,
    Start,
    End,
    Input,
    Output,
    Eq,
    Not,
    Neg,
}

impl From<InputParam> for Node {
    fn from(input: InputParam) -> Self {
        Self::InputPort(input)
    }
}

impl From<OutputParam> for Node {
    fn from(output: OutputParam) -> Self {
        Self::OutputPort(output)
    }
}

impl From<Gamma> for Node {
    fn from(node: Gamma) -> Self {
        Self::Gamma(Box::new(node))
    }
}

impl From<Theta> for Node {
    fn from(node: Theta) -> Self {
        Self::Theta(Box::new(node))
    }
}

impl TryInto<InputParam> for Node {
    type Error = Self;

    fn try_into(self) -> Result<InputParam, Self::Error> {
        if let Self::InputPort(input) = self {
            Ok(input)
        } else {
            Err(self)
        }
    }
}

impl TryInto<InputParam> for &Node {
    type Error = Self;

    fn try_into(self) -> Result<InputParam, Self::Error> {
        if let Node::InputPort(input) = *self {
            Ok(input)
        } else {
            Err(self)
        }
    }
}

impl TryInto<OutputParam> for Node {
    type Error = Self;

    fn try_into(self) -> Result<OutputParam, Self::Error> {
        if let Self::OutputPort(output) = self {
            Ok(output)
        } else {
            Err(self)
        }
    }
}

impl TryInto<OutputParam> for &Node {
    type Error = Self;

    fn try_into(self) -> Result<OutputParam, Self::Error> {
        if let Node::OutputPort(output) = *self {
            Ok(output)
        } else {
            Err(self)
        }
    }
}

impl TryInto<Gamma> for Node {
    type Error = Self;

    fn try_into(self) -> Result<Gamma, Self::Error> {
        if let Self::Gamma(node) = self {
            Ok(*node)
        } else {
            Err(self)
        }
    }
}

impl<'a> TryInto<&'a Gamma> for &'a Node {
    type Error = Self;

    fn try_into(self) -> Result<&'a Gamma, Self::Error> {
        if let Node::Gamma(node) = self {
            Ok(node)
        } else {
            Err(self)
        }
    }
}

impl TryInto<Theta> for Node {
    type Error = Self;

    fn try_into(self) -> Result<Theta, Self::Error> {
        if let Self::Theta(node) = self {
            Ok(*node)
        } else {
            Err(self)
        }
    }
}

impl<'a> TryInto<&'a Theta> for &'a Node {
    type Error = Self;

    fn try_into(self) -> Result<&'a Theta, Self::Error> {
        if let Node::Theta(node) = self {
            Ok(node)
        } else {
            Err(self)
        }
    }
}
