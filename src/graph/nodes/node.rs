use crate::{
    graph::{
        nodes::{
            node_ext::{InputPortKinds, InputPorts, OutputPortKinds, OutputPorts},
            Scan,
        },
        Add, Bool, Byte, EdgeDescriptor, End, Eq, Gamma, Input, InputParam, InputPort, Int, Load,
        Mul, Neg, Neq, NodeExt, NodeId, Not, Output, OutputParam, OutputPort, Start, Store, Sub,
        Theta,
    },
    values::{Cell, Ptr},
};

// TODO: derive_more?
// TODO: Type casting node(s)
// TODO: Byte node?
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Int(Int, Ptr),
    Byte(Byte, Cell),
    Bool(Bool, bool),

    Add(Add),
    Sub(Sub),
    Mul(Mul),

    Eq(Eq),
    Neq(Neq),

    Not(Not),
    Neg(Neg),

    Load(Load),
    Store(Store),
    Scan(Scan),

    Input(Input),
    Output(Output),

    Theta(Box<Theta>),
    Gamma(Box<Gamma>),

    InputParam(InputParam),
    OutputParam(OutputParam),

    Start(Start),
    End(End),
}

node_ext! {
    Int,
    Byte,
    Bool,
    Add,
    Sub,
    Mul,
    Eq,
    Neq,
    Not,
    Neg,
    Load,
    Store,
    Scan,
    Input,
    Output,
    Theta,
    Gamma,
    InputParam,
    OutputParam,
    Start,
    End,
}

impl Node {
    pub fn as_byte_value(&self) -> Option<Cell> {
        if let Self::Byte(_, byte) = *self {
            Some(byte)
        } else {
            None
        }
    }

    pub fn as_int_value(&self) -> Option<Ptr> {
        if let Self::Int(_, int) = *self {
            Some(int)
        } else {
            None
        }
    }

    #[track_caller]
    #[allow(dead_code)]
    pub fn to_int_value(&self) -> Ptr {
        if let Self::Int(_, int) = *self {
            int
        } else {
            panic!("attempted to get int, got {:?}", self);
        }
    }

    pub fn as_bool_value(&self) -> Option<bool> {
        if let Self::Bool(_, bool) = *self {
            Some(bool)
        } else {
            None
        }
    }

    #[track_caller]
    #[allow(dead_code)]
    pub fn to_bool_value(&self) -> bool {
        if let Self::Bool(_, bool) = *self {
            bool
        } else {
            panic!("attempted to get bool, got {:?}", self);
        }
    }

    pub const fn is_const_number(&self) -> bool {
        matches!(self, Self::Byte(..) | Self::Int(..))
    }
}

node_methods! {
    Int,
    Byte,
    Bool,
    Add,
    Sub,
    Mul,
    Load,
    Store,
    Scan,
    Start,
    End,
    Input,
    Output,
    Eq,
    Neq,
    Not,
    Neg,
    InputParam,
    OutputParam,
    Gamma,
    Theta,
}

node_conversions! {
    Add,
    Sub,
    Mul,
    Load,
    Store,
    Scan,
    Start,
    End,
    Input,
    Output,
    Eq,
    Neq,
    Not,
    Neg,
    InputParam,
    OutputParam,
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
