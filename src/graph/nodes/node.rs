use crate::graph::{
    nodes::{
        node_ext::{InputPortKinds, InputPorts, OutputPortKinds},
        ops::Mul,
    },
    Add, Bool, EdgeDescriptor, End, Eq, Gamma, Input, InputParam, InputPort, Int, Load, Neg,
    NodeExt, NodeId, Not, Output, OutputParam, OutputPort, Start, Store, Sub, Theta,
};
use tinyvec::TinyVec;

// TODO: derive_more?
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Int(Int, u32),
    Bool(Bool, bool),
    Add(Add),
    Sub(Sub),
    Mul(Mul),
    Load(Load),
    Store(Store),
    Start(Start),
    End(End),
    Input(Input),
    Output(Output),
    Theta(Box<Theta>),
    InputParam(InputParam),
    OutputParam(OutputParam),
    Eq(Eq),
    Not(Not),
    Neg(Neg),
    Gamma(Box<Gamma>),
}

impl NodeExt for Node {
    fn node(&self) -> NodeId {
        match self {
            Self::Int(int, _) => int.node(),
            Self::Bool(bool, _) => bool.node(),
            Self::Add(add) => add.node(),
            Self::Sub(sub) => sub.node(),
            Self::Mul(mul) => mul.node(),
            Self::Load(load) => load.node(),
            Self::Store(store) => store.node(),
            Self::Start(start) => start.node(),
            Self::End(end) => end.node(),
            Self::Input(input) => input.node(),
            Self::Output(output) => output.node(),
            Self::Theta(theta) => theta.node(),
            Self::InputParam(input_param) => input_param.node(),
            Self::OutputParam(output_param) => output_param.node(),
            Self::Eq(eq) => eq.node(),
            Self::Not(not) => not.node(),
            Self::Neg(neg) => neg.node(),
            Self::Gamma(gamma) => gamma.node(),
        }
    }

    fn input_desc(&self) -> EdgeDescriptor {
        match self {
            Self::Int(int, _) => int.input_desc(),
            Self::Bool(bool, _) => bool.input_desc(),
            Self::Add(add) => add.input_desc(),
            Self::Sub(sub) => sub.input_desc(),
            Self::Mul(mul) => mul.input_desc(),
            Self::Load(load) => load.input_desc(),
            Self::Store(store) => store.input_desc(),
            Self::Start(start) => start.input_desc(),
            Self::End(end) => end.input_desc(),
            Self::Input(input) => input.input_desc(),
            Self::Output(output) => output.input_desc(),
            Self::Theta(theta) => theta.input_desc(),
            Self::InputParam(input_param) => input_param.input_desc(),
            Self::OutputParam(output_param) => output_param.input_desc(),
            Self::Eq(eq) => eq.input_desc(),
            Self::Not(not) => not.input_desc(),
            Self::Neg(neg) => neg.input_desc(),
            Self::Gamma(gamma) => gamma.input_desc(),
        }
    }

    fn all_input_ports(&self) -> InputPorts {
        match self {
            Self::Int(int, _) => int.all_input_ports(),
            Self::Bool(bool, _) => bool.all_input_ports(),
            Self::Add(add) => add.all_input_ports(),
            Self::Sub(sub) => sub.all_input_ports(),
            Self::Mul(mul) => mul.all_input_ports(),
            Self::Load(load) => load.all_input_ports(),
            Self::Store(store) => store.all_input_ports(),
            Self::Start(start) => start.all_input_ports(),
            Self::End(end) => end.all_input_ports(),
            Self::Input(input) => input.all_input_ports(),
            Self::Output(output) => output.all_input_ports(),
            Self::Theta(theta) => theta.all_input_ports(),
            Self::InputParam(input_param) => input_param.all_input_ports(),
            Self::OutputParam(output_param) => output_param.all_input_ports(),
            Self::Eq(eq) => eq.all_input_ports(),
            Self::Not(not) => not.all_input_ports(),
            Self::Neg(neg) => neg.all_input_ports(),
            Self::Gamma(gamma) => gamma.all_input_ports(),
        }
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        match self {
            Self::Int(int, _) => int.all_input_port_kinds(),
            Self::Bool(bool, _) => bool.all_input_port_kinds(),
            Self::Add(add) => add.all_input_port_kinds(),
            Self::Sub(sub) => sub.all_input_port_kinds(),
            Self::Mul(mul) => mul.all_input_port_kinds(),
            Self::Load(load) => load.all_input_port_kinds(),
            Self::Store(store) => store.all_input_port_kinds(),
            Self::Start(start) => start.all_input_port_kinds(),
            Self::End(end) => end.all_input_port_kinds(),
            Self::Input(input) => input.all_input_port_kinds(),
            Self::Output(output) => output.all_input_port_kinds(),
            Self::Theta(theta) => theta.all_input_port_kinds(),
            Self::InputParam(input_param) => input_param.all_input_port_kinds(),
            Self::OutputParam(output_param) => output_param.all_input_port_kinds(),
            Self::Eq(eq) => eq.all_input_port_kinds(),
            Self::Not(not) => not.all_input_port_kinds(),
            Self::Neg(neg) => neg.all_input_port_kinds(),
            Self::Gamma(gamma) => gamma.all_input_port_kinds(),
        }
    }

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        match self {
            Self::Int(int, _) => int.update_input(from, to),
            Self::Bool(bool, _) => bool.update_input(from, to),
            Self::Add(add) => add.update_input(from, to),
            Self::Sub(sub) => sub.update_input(from, to),
            Self::Mul(mul) => mul.update_input(from, to),
            Self::Load(load) => load.update_input(from, to),
            Self::Store(store) => store.update_input(from, to),
            Self::Start(start) => start.update_input(from, to),
            Self::End(end) => end.update_input(from, to),
            Self::Input(input) => input.update_input(from, to),
            Self::Output(output) => output.update_input(from, to),
            Self::Theta(theta) => theta.update_input(from, to),
            Self::InputParam(input_param) => input_param.update_input(from, to),
            Self::OutputParam(output_param) => output_param.update_input(from, to),
            Self::Eq(eq) => eq.update_input(from, to),
            Self::Not(not) => not.update_input(from, to),
            Self::Neg(neg) => neg.update_input(from, to),
            Self::Gamma(gamma) => gamma.update_input(from, to),
        }
    }

    fn output_desc(&self) -> EdgeDescriptor {
        match self {
            Self::Int(int, _) => int.output_desc(),
            Self::Bool(bool, _) => bool.output_desc(),
            Self::Add(add) => add.output_desc(),
            Self::Sub(sub) => sub.output_desc(),
            Self::Mul(mul) => mul.output_desc(),
            Self::Load(load) => load.output_desc(),
            Self::Store(store) => store.output_desc(),
            Self::Start(start) => start.output_desc(),
            Self::End(end) => end.output_desc(),
            Self::Input(input) => input.output_desc(),
            Self::Output(output) => output.output_desc(),
            Self::Theta(theta) => theta.output_desc(),
            Self::InputParam(input_param) => input_param.output_desc(),
            Self::OutputParam(output_param) => output_param.output_desc(),
            Self::Eq(eq) => eq.output_desc(),
            Self::Not(not) => not.output_desc(),
            Self::Neg(neg) => neg.output_desc(),
            Self::Gamma(gamma) => gamma.output_desc(),
        }
    }

    fn all_output_ports(&self) -> TinyVec<[OutputPort; 4]> {
        match self {
            Self::Int(int, _) => int.all_output_ports(),
            Self::Bool(bool, _) => bool.all_output_ports(),
            Self::Add(add) => add.all_output_ports(),
            Self::Sub(sub) => sub.all_output_ports(),
            Self::Mul(mul) => mul.all_output_ports(),
            Self::Load(load) => load.all_output_ports(),
            Self::Store(store) => store.all_output_ports(),
            Self::Start(start) => start.all_output_ports(),
            Self::End(end) => end.all_output_ports(),
            Self::Input(input) => input.all_output_ports(),
            Self::Output(output) => output.all_output_ports(),
            Self::Theta(theta) => theta.all_output_ports(),
            Self::InputParam(input_param) => input_param.all_output_ports(),
            Self::OutputParam(output_param) => output_param.all_output_ports(),
            Self::Eq(eq) => eq.all_output_ports(),
            Self::Not(not) => not.all_output_ports(),
            Self::Neg(neg) => neg.all_output_ports(),
            Self::Gamma(gamma) => gamma.all_output_ports(),
        }
    }

    fn all_output_port_kinds(&self) -> OutputPortKinds {
        match self {
            Self::Int(int, _) => int.all_output_port_kinds(),
            Self::Bool(bool, _) => bool.all_output_port_kinds(),
            Self::Add(add) => add.all_output_port_kinds(),
            Self::Sub(sub) => sub.all_output_port_kinds(),
            Self::Mul(mul) => mul.all_output_port_kinds(),
            Self::Load(load) => load.all_output_port_kinds(),
            Self::Store(store) => store.all_output_port_kinds(),
            Self::Start(start) => start.all_output_port_kinds(),
            Self::End(end) => end.all_output_port_kinds(),
            Self::Input(input) => input.all_output_port_kinds(),
            Self::Output(output) => output.all_output_port_kinds(),
            Self::Theta(theta) => theta.all_output_port_kinds(),
            Self::InputParam(input_param) => input_param.all_output_port_kinds(),
            Self::OutputParam(output_param) => output_param.all_output_port_kinds(),
            Self::Eq(eq) => eq.all_output_port_kinds(),
            Self::Not(not) => not.all_output_port_kinds(),
            Self::Neg(neg) => neg.all_output_port_kinds(),
            Self::Gamma(gamma) => gamma.all_output_port_kinds(),
        }
    }

    fn update_output(&mut self, from: OutputPort, to: OutputPort) {
        match self {
            Self::Int(int, _) => int.update_output(from, to),
            Self::Bool(bool, _) => bool.update_output(from, to),
            Self::Add(add) => add.update_output(from, to),
            Self::Sub(sub) => sub.update_output(from, to),
            Self::Mul(mul) => mul.update_output(from, to),
            Self::Load(load) => load.update_output(from, to),
            Self::Store(store) => store.update_output(from, to),
            Self::Start(start) => start.update_output(from, to),
            Self::End(end) => end.update_output(from, to),
            Self::Input(input) => input.update_output(from, to),
            Self::Output(output) => output.update_output(from, to),
            Self::Theta(theta) => theta.update_output(from, to),
            Self::InputParam(input_param) => input_param.update_output(from, to),
            Self::OutputParam(output_param) => output_param.update_output(from, to),
            Self::Eq(eq) => eq.update_output(from, to),
            Self::Not(not) => not.update_output(from, to),
            Self::Neg(neg) => neg.update_output(from, to),
            Self::Gamma(gamma) => gamma.update_output(from, to),
        }
    }
}

impl Node {
    /// Returns `true` if the node is [`Int`].
    ///
    /// [`Int`]: Node::Int
    pub const fn is_int(&self) -> bool {
        matches!(self, Self::Int(..))
    }

    /// Returns `true` if the node is a [`Theta`].
    ///
    /// [`Theta`]: Node::Theta
    pub const fn is_theta(&self) -> bool {
        matches!(self, Self::Theta(..))
    }

    /// Returns `true` if the node is a [`Gamma`].
    ///
    /// [`Gamma`]: Node::Gamma
    pub const fn is_gamma(&self) -> bool {
        matches!(self, Self::Gamma(..))
    }

    pub const fn as_int(&self) -> Option<(Int, u32)> {
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

    pub const fn as_theta(&self) -> Option<&Theta> {
        if let Self::Theta(theta) = self {
            Some(theta)
        } else {
            None
        }
    }

    pub fn as_theta_mut(&mut self) -> Option<&mut Theta> {
        if let Self::Theta(theta) = self {
            Some(theta)
        } else {
            None
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
    pub fn to_int(&self) -> Int {
        if let Self::Int(int, _) = *self {
            int
        } else {
            panic!("attempted to get int, got {:?}", self);
        }
    }

    #[track_caller]
    #[allow(dead_code)]
    pub fn to_int_value(&self) -> u32 {
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
    #[allow(dead_code)]
    pub fn to_theta_mut(&mut self) -> &mut Theta {
        if let Self::Theta(theta) = self {
            theta
        } else {
            panic!("attempted to get theta, got {:?}", self);
        }
    }
}

macro_rules! node_variants {
    ($($type:ident $(as $name:ident)?),* $(,)?) => {
        use paste::paste;

        $(
            impl From<$type> for Node {
                fn from(node: $type) -> Self {
                    node_variants!(@variant node, $type, $($name)?)
                }
            }

            impl TryInto<$type> for Node {
                type Error = Self;

                fn try_into(self) -> Result<$type, Self::Error> {
                    if let node_variants!(@variant node, $type, $($name)?) = self {
                        Ok(node)
                    } else {
                        Err(self)
                    }
                }
            }

            impl TryInto<$type> for &Node {
                type Error = Self;

                fn try_into(self) -> Result<$type, Self::Error> {
                    if let node_variants!(@variant node, $type, $($name)?) = *self {
                        Ok(node)
                    } else {
                        Err(self)
                    }
                }
            }

            impl<'a> TryInto<&'a $type> for &'a Node {
                type Error = Self;

                fn try_into(self) -> Result<&'a $type, Self::Error> {
                    if let node_variants!(@variant node, $type, $($name)?) = self {
                        Ok(node)
                    } else {
                        Err(self)
                    }
                }
            }

            impl Node {
                paste! {
                    pub const fn [<is_ $type:snake>](&self) -> bool {
                        matches!(self, node_variants!(@pat $type, $($name)?))
                    }

                    pub const fn [<as_ $type:snake>](&self) -> Option<$type> {
                        if let node_variants!(@variant node, $type, $($name)?) = *self {
                            Some(node)
                        } else {
                            None
                        }
                    }

                    pub fn [<as_ $type:snake _mut>](&mut self) -> Option<&mut $type> {
                        if let node_variants!(@variant node, $type, $($name)?) = self {
                            Some(node)
                        } else {
                            None
                        }
                    }

                    #[track_caller]
                    pub fn [<to_ $type:snake>](&self) -> $type {
                        if let node_variants!(@variant node, $type, $($name)?) = *self {
                            node
                        } else {
                            panic!(
                                concat!("attempted to get", stringify!($type), " got {:?}"),
                                self,
                            );
                        }
                    }

                    #[track_caller]
                    pub fn [<to_ $type:snake _mut>](&mut self) -> &mut $type {
                        if let node_variants!(@variant node, $type, $($name)?) = self {
                            node
                        } else {
                            panic!(
                                concat!("attempted to get", stringify!($type), " got {:?}"),
                                self,
                            );
                        }
                    }
                }
            }
        )*
    };

    (@variant $inner:ident, $variant_type:ident, $variant_name:ident $(,)?) => { Node::$variant_name($inner) };
    (@variant $inner:ident, $variant_type:ident $(,)?) => { Node::$variant_type($inner) };

    (@pat $variant_type:ident, $variant_name:ident $(,)?) => { Node::$variant_name(_) };
    (@pat $variant_type:ident $(,)?) => { Node::$variant_type(_) };
}

node_variants! {
    Add,
    Sub,
    Mul,
    Load,
    Store,
    Start,
    End,
    Input,
    Output,
    Eq,
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
