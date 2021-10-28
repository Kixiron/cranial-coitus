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
    InputParam(InputParam),
    OutputParam(OutputParam),
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
            | Self::Input(Input { node, .. })
            | Self::Output(Output { node, .. })
            | Self::Eq(Eq { node, .. })
            | Self::Not(Not { node, .. })
            | Self::Neg(Neg { node, .. }) => *node,
            Self::Start(start) => start.node(),
            Self::End(end) => end.node(),
            Self::InputParam(input_param) => input_param.node(),
            Self::OutputParam(output_param) => output_param.node(),
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
            Self::Start(start) => start.all_input_ports(),
            Self::End(end) => end.all_input_ports(),
            Self::Input(input) => tiny_vec![input.effect_in],
            Self::Output(output) => tiny_vec![output.value, output.effect_in],
            Self::Theta(theta) => theta.all_input_ports(),
            Self::InputParam(input_param) => input_param.all_input_ports(),
            Self::OutputParam(output_param) => output_param.all_input_ports(),
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
    pub fn inputs_mut(&mut self) -> Vec<(&mut InputPort, EdgeKind)> {
        match self {
            Self::Int(_, _) | Self::Bool(_, _) => Vec::new(),
            Self::Add(add) => vec![
                (&mut add.lhs, EdgeKind::Value),
                (&mut add.rhs, EdgeKind::Value),
            ],
            Self::Load(load) => vec![
                (&mut load.ptr, EdgeKind::Value),
                (&mut load.effect_in, EdgeKind::Effect),
            ],
            Self::Store(store) => vec![
                (&mut store.ptr, EdgeKind::Value),
                (&mut store.value, EdgeKind::Value),
                (&mut store.effect_in, EdgeKind::Effect),
            ],
            Self::Start(_) => Vec::new(),
            Self::End(end) => vec![(&mut end.input_effect, EdgeKind::Effect)],
            Self::Input(input) => vec![(&mut input.effect_in, EdgeKind::Effect)],
            Self::Output(output) => {
                vec![
                    (&mut output.value, EdgeKind::Value),
                    (&mut output.effect_in, EdgeKind::Effect),
                ]
            }
            Self::Theta(_) => {
                tracing::warn!("use setters for setting inputs on theta nodes");
                Vec::new()
            }
            Self::InputParam(_) => Vec::new(),
            Self::OutputParam(output) => vec![(&mut output.input, EdgeKind::Value)],
            Self::Eq(eq) => vec![
                (&mut eq.lhs, EdgeKind::Value),
                (&mut eq.rhs, EdgeKind::Value),
            ],
            Self::Not(not) => vec![(&mut not.input, EdgeKind::Value)],
            Self::Neg(neg) => vec![(&mut neg.input, EdgeKind::Value)],
            Self::Gamma(gamma) => {
                let mut inputs: Vec<_> = gamma
                    .inputs
                    .iter_mut()
                    .map(|input| (input, EdgeKind::Value))
                    .collect();
                inputs.push((&mut gamma.condition, EdgeKind::Value));
                inputs.push((&mut gamma.effect_in, EdgeKind::Effect));
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
            Self::Start(start) => start.all_output_ports(),
            Self::End(end) => end.all_output_ports(),
            Self::Input(input) => tiny_vec![input.value, input.effect_out],
            Self::Output(output) => tiny_vec![output.effect_out],
            Self::Theta(theta) => theta.all_output_ports(),
            Self::InputParam(input_param) => input_param.all_output_ports(),
            Self::OutputParam(output_param) => output_param.all_output_ports(),
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
    pub fn outputs_mut(&mut self) -> Vec<(&mut OutputPort, EdgeKind)> {
        match self {
            Self::Int(int, _) => vec![(&mut int.value, EdgeKind::Value)],
            Self::Bool(bool, _) => vec![(&mut bool.value, EdgeKind::Value)],
            Self::Add(add) => vec![(&mut add.value, EdgeKind::Value)],
            Self::Load(load) => vec![
                (&mut load.value, EdgeKind::Value),
                (&mut load.effect_out, EdgeKind::Effect),
            ],
            Self::Store(store) => vec![(&mut store.effect_out, EdgeKind::Effect)],
            Self::Start(start) => vec![(&mut start.effect, EdgeKind::Effect)],
            Self::End(_) => Vec::new(),
            Self::Input(input) => vec![
                (&mut input.value, EdgeKind::Value),
                (&mut input.effect_out, EdgeKind::Effect),
            ],
            Self::Output(output) => vec![(&mut output.effect_out, EdgeKind::Effect)],
            Self::Theta(_) => {
                tracing::warn!("use setters for setting outputs on theta nodes");
                Vec::new()
            }
            Self::InputParam(input) => vec![(&mut input.output, EdgeKind::Value)],
            Self::OutputParam(_) => Vec::new(),
            Self::Eq(eq) => vec![(&mut eq.value, EdgeKind::Value)],
            Self::Not(not) => vec![(&mut not.value, EdgeKind::Value)],
            Self::Neg(neg) => vec![(&mut neg.value, EdgeKind::Value)],
            Self::Gamma(gamma) => {
                let mut outputs: Vec<_> = gamma
                    .outputs
                    .iter_mut()
                    .map(|output| (output, EdgeKind::Value))
                    .collect();
                outputs.push((&mut gamma.effect_out, EdgeKind::Effect));
                outputs
            }
        }
    }

    pub fn input_desc(&self) -> EdgeDescriptor {
        match self {
            Self::Int(..) | Self::Bool(..) | Self::InputParam(_) => {
                EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::zero())
            }
            Self::Add(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::two()),
            Self::Load(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::one()),
            Self::Store(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::two()),
            Self::Start(start) => start.input_desc(),
            Self::End(end) => end.input_desc(),
            Self::Input(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
            Self::Output(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::one()),
            Self::Theta(theta) => theta.input_desc(),
            Self::OutputParam(output) => match output.kind {
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
            Self::Int(..) | Self::Bool(..) | Self::Add(_) | Self::OutputParam(_) => {
                EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::unlimited())
            }
            Self::Load(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::unlimited()),
            Self::Store(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
            Self::Start(start) => start.output_desc(),
            Self::End(end) => end.output_desc(),
            Self::Input(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::one()),
            Self::Output(_) => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
            Self::Theta(theta) => theta.output_desc(),
            Self::InputParam(output) => match output.kind {
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
    #[allow(dead_code)]
    pub fn to_theta_mut(&mut self) -> &mut Theta {
        if let Self::Theta(theta) = self {
            theta
        } else {
            panic!("attempted to get theta, got {:?}", self);
        }
    }

    /// Returns `true` if the node is a [`Theta`].
    ///
    /// [`Theta`]: Node::Theta
    pub const fn is_theta(&self) -> bool {
        matches!(self, Self::Theta(..))
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
