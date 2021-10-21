use crate::graph::{
    Add, Bool, EdgeCount, EdgeDescriptor, EdgeKind, End, Eq, Input, InputParam, InputPort, Int,
    Load, Neg, NodeId, Not, Output, OutputParam, OutputPort, Phi, Start, Store, Theta,
};

#[derive(Debug, Clone)]
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
    Theta(Theta),
    InputPort(InputParam),
    OutputPort(OutputParam),
    Eq(Eq),
    Not(Not),
    Neg(Neg),
    Phi(Phi),
}

impl Node {
    pub fn node_id(&self) -> NodeId {
        match *self {
            Self::Int(Int { node, .. }, _)
            | Self::Bool(Bool { node, .. }, _)
            | Self::Add(Add { node, .. })
            | Self::Load(Load { node, .. })
            | Self::Store(Store { node, .. })
            | Self::Start(Start { node, .. })
            | Self::End(End { node, .. })
            | Self::Input(Input { node, .. })
            | Self::Output(Output { node, .. })
            | Self::Theta(Theta { node, .. })
            | Self::InputPort(InputParam { node, .. })
            | Self::OutputPort(OutputParam { node, .. })
            | Self::Eq(Eq { node, .. })
            | Self::Not(Not { node, .. })
            | Self::Neg(Neg { node, .. })
            | Self::Phi(Phi { node, .. }) => node,
        }
    }

    // FIXME: TinyVec?
    pub fn inputs(&self) -> Vec<InputPort> {
        match self {
            Self::Int(_, _) | Self::Bool(_, _) => Vec::new(),
            Self::Add(add) => vec![add.lhs, add.rhs],
            Self::Load(load) => vec![load.ptr, load.effect_in],
            Self::Store(store) => vec![store.ptr, store.value, store.effect_in],
            Self::Start(_) => Vec::new(),
            Self::End(end) => vec![end.effect],
            Self::Input(input) => vec![input.effect_in],
            Self::Output(output) => vec![output.value, output.effect_in],
            Self::Theta(theta) => {
                let mut inputs = theta.inputs.to_vec();
                inputs.push(theta.effect_in);
                inputs
            }
            Self::InputPort(_) => Vec::new(),
            Self::OutputPort(output) => vec![output.value],
            Self::Eq(eq) => vec![eq.lhs, eq.rhs],
            Self::Not(not) => vec![not.input],
            Self::Neg(neg) => vec![neg.input],
            Self::Phi(phi) => {
                let mut inputs = phi.inputs.to_vec();
                inputs.push(phi.condition);
                inputs.push(phi.effect_in);
                inputs
            }
        }
    }

    // FIXME: TinyVec?
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
            Self::Theta(theta) => {
                let mut inputs: Vec<_> = theta.inputs.iter_mut().collect();
                inputs.push(&mut theta.effect_in);
                inputs
            }
            Self::InputPort(_) => Vec::new(),
            Self::OutputPort(output) => vec![&mut output.value],
            Self::Eq(eq) => vec![&mut eq.lhs, &mut eq.rhs],
            Self::Not(not) => vec![&mut not.input],
            Self::Neg(neg) => vec![&mut neg.input],
            Self::Phi(phi) => {
                let mut inputs: Vec<_> = phi.inputs.iter_mut().collect();
                inputs.push(&mut phi.condition);
                inputs.push(&mut phi.effect_in);
                inputs
            }
        }
    }

    // FIXME: TinyVec?
    pub fn outputs(&self) -> Vec<OutputPort> {
        match self {
            Self::Int(int, _) => vec![int.value],
            Self::Bool(bool, _) => vec![bool.value],
            Self::Add(add) => vec![add.value],
            Self::Load(load) => vec![load.value, load.effect_out],
            Self::Store(store) => vec![store.effect_out],
            Self::Start(start) => vec![start.effect],
            Self::End(_) => Vec::new(),
            Self::Input(input) => vec![input.value, input.effect_out],
            Self::Output(output) => vec![output.effect_out],
            Self::Theta(theta) => {
                let mut inputs = theta.outputs.to_vec();
                inputs.push(theta.effect_out);
                inputs
            }
            Self::InputPort(input) => vec![input.value],
            Self::OutputPort(_) => Vec::new(),
            Self::Eq(eq) => vec![eq.value],
            Self::Not(not) => vec![not.value],
            Self::Neg(neg) => vec![neg.value],
            Self::Phi(phi) => {
                let mut inputs = phi.outputs.to_vec();
                inputs.push(phi.effect_out);
                inputs
            }
        }
    }

    // FIXME: TinyVec?
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
            Self::Theta(theta) => {
                let mut inputs: Vec<_> = theta.outputs.iter_mut().collect();
                inputs.push(&mut theta.effect_out);
                inputs
            }
            Self::InputPort(input) => vec![&mut input.value],
            Self::OutputPort(_) => Vec::new(),
            Self::Eq(eq) => vec![&mut eq.value],
            Self::Not(not) => vec![&mut not.value],
            Self::Neg(neg) => vec![&mut neg.value],
            Self::Phi(phi) => {
                let mut inputs: Vec<_> = phi.outputs.iter_mut().collect();
                inputs.push(&mut phi.effect_out);
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
            Self::Theta(theta) => {
                EdgeDescriptor::new(EdgeCount::one(), EdgeCount::exact(theta.inputs().len()))
            }
            Self::OutputPort(output) => match output.kind {
                EdgeKind::Effect => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
                EdgeKind::Value => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
            },
            Self::Eq(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::two()),
            Self::Not(_) | Self::Neg(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
            Self::Phi(phi) => {
                EdgeDescriptor::new(EdgeCount::one(), EdgeCount::exact(phi.inputs().len() + 1))
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
            Self::Theta(theta) => EdgeDescriptor::new(
                EdgeCount::one(),
                EdgeCount::new(None, Some(theta.outputs().len())),
            ),
            Self::InputPort(output) => match output.kind {
                EdgeKind::Effect => EdgeDescriptor::new(EdgeCount::one(), EdgeCount::zero()),
                EdgeKind::Value => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
            },
            Self::Eq(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
            Self::Not(_) | Self::Neg(_) => EdgeDescriptor::new(EdgeCount::zero(), EdgeCount::one()),
            Self::Phi(phi) => {
                EdgeDescriptor::new(EdgeCount::one(), EdgeCount::exact(phi.outputs().len()))
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
    pub fn to_phi_mut(&mut self) -> &mut Phi {
        if let Self::Phi(phi) = self {
            phi
        } else {
            panic!("attempted to get phi, got {:?}", self);
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

impl From<InputParam> for Node {
    fn from(input: InputParam) -> Self {
        Self::InputPort(input)
    }
}

impl From<Theta> for Node {
    fn from(theta: Theta) -> Self {
        Self::Theta(theta)
    }
}

impl From<Output> for Node {
    fn from(output: Output) -> Self {
        Self::Output(output)
    }
}

impl From<Input> for Node {
    fn from(input: Input) -> Self {
        Self::Input(input)
    }
}

impl From<End> for Node {
    fn from(end: End) -> Self {
        Self::End(end)
    }
}

impl From<Start> for Node {
    fn from(start: Start) -> Self {
        Self::Start(start)
    }
}

impl From<Store> for Node {
    fn from(store: Store) -> Self {
        Self::Store(store)
    }
}

impl From<Load> for Node {
    fn from(load: Load) -> Self {
        Self::Load(load)
    }
}

impl From<Add> for Node {
    fn from(add: Add) -> Self {
        Self::Add(add)
    }
}

impl From<OutputParam> for Node {
    fn from(output: OutputParam) -> Self {
        Self::OutputPort(output)
    }
}

impl From<Eq> for Node {
    fn from(eq: Eq) -> Self {
        Self::Eq(eq)
    }
}

impl From<Not> for Node {
    fn from(not: Not) -> Self {
        Self::Not(not)
    }
}

impl From<Phi> for Node {
    fn from(phi: Phi) -> Self {
        Self::Phi(phi)
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

impl TryInto<Load> for Node {
    type Error = Self;

    fn try_into(self) -> Result<Load, Self::Error> {
        if let Self::Load(load) = self {
            Ok(load)
        } else {
            Err(self)
        }
    }
}

impl TryInto<Load> for &Node {
    type Error = Self;

    fn try_into(self) -> Result<Load, Self::Error> {
        if let Node::Load(load) = *self {
            Ok(load)
        } else {
            Err(self)
        }
    }
}

impl TryInto<Store> for Node {
    type Error = Self;

    fn try_into(self) -> Result<Store, Self::Error> {
        if let Self::Store(store) = self {
            Ok(store)
        } else {
            Err(self)
        }
    }
}

impl TryInto<Store> for &Node {
    type Error = Self;

    fn try_into(self) -> Result<Store, Self::Error> {
        if let Node::Store(store) = *self {
            Ok(store)
        } else {
            Err(self)
        }
    }
}

impl TryInto<Add> for Node {
    type Error = Self;

    fn try_into(self) -> Result<Add, Self::Error> {
        if let Self::Add(add) = self {
            Ok(add)
        } else {
            Err(self)
        }
    }
}

impl TryInto<Add> for &Node {
    type Error = Self;

    fn try_into(self) -> Result<Add, Self::Error> {
        if let Node::Add(add) = *self {
            Ok(add)
        } else {
            Err(self)
        }
    }
}

impl TryInto<Eq> for Node {
    type Error = Self;

    fn try_into(self) -> Result<Eq, Self::Error> {
        if let Self::Eq(eq) = self {
            Ok(eq)
        } else {
            Err(self)
        }
    }
}

impl TryInto<Eq> for &Node {
    type Error = Self;

    fn try_into(self) -> Result<Eq, Self::Error> {
        if let Node::Eq(eq) = *self {
            Ok(eq)
        } else {
            Err(self)
        }
    }
}

impl TryInto<Not> for Node {
    type Error = Self;

    fn try_into(self) -> Result<Not, Self::Error> {
        if let Self::Not(not) = self {
            Ok(not)
        } else {
            Err(self)
        }
    }
}

impl TryInto<Not> for &Node {
    type Error = Self;

    fn try_into(self) -> Result<Not, Self::Error> {
        if let Node::Not(not) = *self {
            Ok(not)
        } else {
            Err(self)
        }
    }
}

impl TryInto<Start> for Node {
    type Error = Self;

    fn try_into(self) -> Result<Start, Self::Error> {
        if let Self::Start(start) = self {
            Ok(start)
        } else {
            Err(self)
        }
    }
}

impl TryInto<Start> for &Node {
    type Error = Self;

    fn try_into(self) -> Result<Start, Self::Error> {
        if let Node::Start(start) = *self {
            Ok(start)
        } else {
            Err(self)
        }
    }
}

impl TryInto<End> for Node {
    type Error = Self;

    fn try_into(self) -> Result<End, Self::Error> {
        if let Self::End(end) = self {
            Ok(end)
        } else {
            Err(self)
        }
    }
}

impl TryInto<End> for &Node {
    type Error = Self;

    fn try_into(self) -> Result<End, Self::Error> {
        if let Node::End(end) = *self {
            Ok(end)
        } else {
            Err(self)
        }
    }
}
