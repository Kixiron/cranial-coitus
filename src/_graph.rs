use std::{
    cell::Cell,
    collections::HashMap,
    fmt::{self, Debug, Write as _},
    rc::Rc,
};

#[derive(Debug)]
pub struct Rvsdg {
    // TODO: Ports
    pub nodes: HashMap<NodeId, Node>,
    pub edges_forward: HashMap<NodeId, Vec<(NodeId, Edge)>>,
    pub edges_reverse: HashMap<NodeId, Vec<(NodeId, Edge)>>,
    pub node_counter: Rc<Cell<NodeId>>,
}

impl Rvsdg {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges_forward: HashMap::new(),
            edges_reverse: HashMap::new(),
            node_counter: Rc::new(Cell::new(NodeId(0))),
        }
    }

    pub fn with_counters(node_counter: Rc<Cell<NodeId>>) -> Self {
        Self {
            nodes: HashMap::new(),
            edges_forward: HashMap::new(),
            edges_reverse: HashMap::new(),
            node_counter,
        }
    }

    fn next_node(&self) -> NodeId {
        let id = self.node_counter.get();
        self.node_counter.set(NodeId(id.0 + 1));

        id
    }

    pub fn add_node<N>(&mut self, node: N) -> NodeId
    where
        N: Into<Node>,
    {
        let id = self.next_node();
        let result = self.nodes.insert(id, node.into());
        assert!(result.is_none(), "double-inserted a node");

        id
    }

    pub fn add_edge(&mut self, src: NodeId, dest: NodeId, edge: Edge) {
        // Insert the edges
        self.edges_forward
            .entry(src)
            .or_default()
            .push((dest, edge));

        self.edges_reverse
            .entry(dest)
            .or_default()
            .push((src, edge));
    }

    pub fn inputs(&self, node: NodeId) -> &[(NodeId, Edge)] {
        self.edges_reverse
            .get(&node)
            .map(|edges| &**edges)
            .unwrap_or(&[])
    }

    pub fn outputs(&self, node: NodeId) -> &[(NodeId, Edge)] {
        self.edges_forward
            .get(&node)
            .map(|edges| &**edges)
            .unwrap_or(&[])
    }

    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes.iter().map(|(&id, node)| (id, node))
    }

    #[track_caller]
    pub fn node(&self, node: NodeId) -> &Node {
        self.nodes
            .get(&node)
            .expect("attempted to get node that doesn't exist")
    }

    #[track_caller]
    pub fn node_mut(&mut self, node: NodeId) -> &mut Node {
        self.nodes
            .get_mut(&node)
            .expect("attempted to get node that doesn't exist")
    }

    pub fn load(&mut self, last_effect: &mut NodeId, ptr: NodeId) -> NodeId {
        // Create the load node
        let load = self.add_node(Load::new(ptr, *last_effect));

        // Create the value edge from the pointer to the load
        self.add_edge(ptr, load, Edge::Value);

        // Create the state edge between the last effect and load
        self.add_edge(*last_effect, load, Edge::Effect);
        *last_effect = load;

        load
    }

    pub fn store(&mut self, last_effect: &mut NodeId, ptr: NodeId, value: NodeId) -> NodeId {
        // Create the store node
        let store = self.add_node(Store::new(ptr, value, *last_effect));

        // Create the state edge between the last effect and store
        self.add_edge(*last_effect, store, Edge::Effect);
        *last_effect = store;

        // Create a value edge from the pointer to the store
        self.add_edge(ptr, store, Edge::Value);
        // Create a value edge from the value to the store
        self.add_edge(value, store, Edge::Value);

        store
    }

    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        // Create the add node
        let add = self.add_node(Add::new(lhs, rhs));

        // Create the value edges between the operands and the add node
        self.add_edge(lhs, add, Edge::Value);
        self.add_edge(rhs, add, Edge::Value);

        add
    }

    pub fn eq(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        // Create the eq node
        let eq = self.add_node(Eq::new(lhs, rhs));

        // Create the value edges between the operands and the eq node
        self.add_edge(lhs, eq, Edge::Value);
        self.add_edge(rhs, eq, Edge::Value);

        eq
    }

    pub fn read(&mut self, last_effect: &mut NodeId) -> NodeId {
        // Create the read node
        let read = self.add_node(Read::new(*last_effect));

        // Create the effect edge between the last effect and the read
        self.add_edge(*last_effect, read, Edge::Effect);
        *last_effect = read;

        read
    }

    pub fn write(&mut self, last_effect: &mut NodeId, value: NodeId) -> NodeId {
        // Create the read node
        let write = self.add_node(Write::new(value, *last_effect));

        // Create the value edge for the written value
        self.add_edge(value, write, Edge::Value);

        // Create the effect edge between the last effect and the write
        self.add_edge(*last_effect, write, Edge::Effect);
        *last_effect = write;

        write
    }

    pub fn phi<T, F>(
        &mut self,
        last_effect: &mut NodeId,
        cond: NodeId,
        truthy: T,
        falsy: F,
    ) -> NodeId
    where
        T: FnOnce(&mut NodeId, &mut Rvsdg),
        F: FnOnce(&mut NodeId, &mut Rvsdg),
    {
        // Create the phi node
        let phi = self.add_node(Phi::new(
            cond,
            *last_effect,
            Box::new([
                Rvsdg::with_counters(self.node_counter.clone()),
                Rvsdg::with_counters(self.node_counter.clone()),
            ]),
        ));

        // Create the condition edge
        self.add_edge(cond, phi, Edge::Value);

        // Create the effect edge into the phi node
        self.add_edge(*last_effect, phi, Edge::Effect);
        *last_effect = phi;

        let phi_node = self.node_mut(phi).to_phi_mut();

        // Build the two branch's subgraphs
        truthy(last_effect, &mut phi_node.branches[0]);
        falsy(last_effect, &mut phi_node.branches[1]);

        phi
    }

    pub fn theta<T>(&mut self, last_effect: &mut NodeId, body: T) -> NodeId
    where
        T: FnOnce(&mut NodeId, &mut Rvsdg) -> NodeId,
    {
        // Create the theta node
        let theta = self.add_node(Theta::new(
            NodeId::MAX,
            *last_effect,
            Box::new(Rvsdg::with_counters(self.node_counter.clone())),
        ));

        // Create the effect edge into the theta node
        self.add_edge(*last_effect, theta, Edge::Effect);
        *last_effect = theta;

        let theta_node = self.node_mut(theta).to_theta_mut();

        // Build the theta's subgraph
        let cond = body(last_effect, &mut theta_node.body);

        // Add the condition to the theta
        theta_node.cond = cond;

        theta
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct NodeId(u32);

impl NodeId {
    const MAX: Self = Self(u32::MAX);
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NodeId(")?;
        Debug::fmt(&self.0, f)?;
        f.write_char(')')
    }
}

#[derive(Debug)]
pub enum Node {
    Add(Add),
    Load(Load),
    Store(Store),
    Theta(Theta),
    Int(i32),
    Bool(bool),
    Array(Array),
    Start(Start),
    End(End),
    Read(Read),
    Write(Write),
    Eq(Eq),
    Phi(Phi),
}

impl Node {
    pub fn as_add_mut(&mut self) -> Option<&mut Add> {
        if let Self::Add(add) = self {
            Some(add)
        } else {
            None
        }
    }

    #[track_caller]
    pub fn to_add_mut(&mut self) -> &mut Add {
        self.as_add_mut()
            .expect("attempted to cast a node to an add")
    }

    pub fn as_store_mut(&mut self) -> Option<&mut Store> {
        if let Self::Store(store) = self {
            Some(store)
        } else {
            None
        }
    }

    #[track_caller]
    pub fn to_store_mut(&mut self) -> &mut Store {
        self.as_store_mut()
            .expect("attempted to cast a node to a store")
    }

    pub fn as_write_mut(&mut self) -> Option<&mut Write> {
        if let Self::Write(write) = self {
            Some(write)
        } else {
            None
        }
    }

    #[track_caller]
    pub fn to_write_mut(&mut self) -> &mut Write {
        self.as_write_mut()
            .expect("attempted to cast a node to a write")
    }

    pub fn as_eq_mut(&mut self) -> Option<&mut Eq> {
        if let Self::Eq(eq) = self {
            Some(eq)
        } else {
            None
        }
    }

    #[track_caller]
    pub fn to_eq_mut(&mut self) -> &mut Eq {
        self.as_eq_mut().expect("attempted to cast a node to an eq")
    }

    pub fn as_phi_mut(&mut self) -> Option<&mut Phi> {
        if let Self::Phi(phi) = self {
            Some(phi)
        } else {
            None
        }
    }

    #[track_caller]
    pub fn to_phi_mut(&mut self) -> &mut Phi {
        self.as_phi_mut()
            .expect("attempted to cast a node to a phi")
    }

    pub fn as_theta_mut(&mut self) -> Option<&mut Theta> {
        if let Self::Theta(theta) = self {
            Some(theta)
        } else {
            None
        }
    }

    #[track_caller]
    pub fn to_theta_mut(&mut self) -> &mut Theta {
        self.as_theta_mut()
            .expect("attempted to cast a node to a theta")
    }
}

impl From<i32> for Node {
    fn from(int: i32) -> Self {
        Self::Int(int)
    }
}

impl From<bool> for Node {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

#[derive(Debug)]
pub struct Start;

impl From<Start> for Node {
    fn from(start: Start) -> Self {
        Self::Start(start)
    }
}

#[derive(Debug)]
pub struct End {
    effect: NodeId,
}

impl End {
    pub fn new(effect: NodeId) -> Self {
        Self { effect }
    }
}

impl From<End> for Node {
    fn from(end: End) -> Self {
        Self::End(end)
    }
}

#[derive(Debug)]
pub struct Load {
    pub ptr: NodeId,
    pub effect: NodeId,
}

impl Load {
    pub fn new(ptr: NodeId, effect: NodeId) -> Self {
        Self { ptr, effect }
    }
}

impl From<Load> for Node {
    fn from(load: Load) -> Self {
        Self::Load(load)
    }
}

/// Reads a single byte from stdin
#[derive(Debug)]
pub struct Read {
    pub effect: NodeId,
}

impl Read {
    pub fn new(effect: NodeId) -> Self {
        Self { effect }
    }
}

impl From<Read> for Node {
    fn from(read: Read) -> Self {
        Self::Read(read)
    }
}

/// Writes a single byte to stdin
#[derive(Debug)]
pub struct Write {
    pub value: NodeId,
    pub effect: NodeId,
}

impl Write {
    pub fn new(value: NodeId, effect: NodeId) -> Self {
        Self { value, effect }
    }
}

impl From<Write> for Node {
    fn from(write: Write) -> Self {
        Self::Write(write)
    }
}

#[derive(Debug)]
pub struct Store {
    pub ptr: NodeId,
    pub value: NodeId,
    pub effect: NodeId,
}

impl Store {
    pub fn new(ptr: NodeId, value: NodeId, effect: NodeId) -> Self {
        Self { ptr, value, effect }
    }
}

impl From<Store> for Node {
    fn from(store: Store) -> Self {
        Self::Store(store)
    }
}

#[derive(Debug)]
pub struct Add {
    pub lhs: NodeId,
    pub rhs: NodeId,
}

impl Add {
    pub fn new(lhs: NodeId, rhs: NodeId) -> Self {
        Self { lhs, rhs }
    }
}

impl From<Add> for Node {
    fn from(add: Add) -> Self {
        Self::Add(add)
    }
}

#[derive(Debug)]
pub struct Eq {
    pub lhs: NodeId,
    pub rhs: NodeId,
}

impl Eq {
    pub fn new(lhs: NodeId, rhs: NodeId) -> Self {
        Self { lhs, rhs }
    }
}

impl From<Eq> for Node {
    fn from(eq: Eq) -> Self {
        Self::Eq(eq)
    }
}

#[derive(Debug)]
pub struct Theta {
    pub cond: NodeId,
    pub effect: NodeId,
    pub body: Box<Rvsdg>,
}

impl Theta {
    pub fn new(cond: NodeId, effect: NodeId, body: Box<Rvsdg>) -> Self {
        Self { cond, effect, body }
    }
}

impl From<Theta> for Node {
    fn from(theta: Theta) -> Self {
        Self::Theta(theta)
    }
}

#[derive(Debug)]
pub struct Phi {
    pub cond: NodeId,
    pub effect: NodeId,
    pub branches: Box<[Rvsdg; 2]>,
}

impl Phi {
    pub fn new(cond: NodeId, effect: NodeId, branches: Box<[Rvsdg; 2]>) -> Self {
        Self {
            cond,
            effect,
            branches,
        }
    }
}

impl From<Phi> for Node {
    fn from(phi: Phi) -> Self {
        Self::Phi(phi)
    }
}

#[derive(Debug)]
pub struct Array {
    pub values: Vec<NodeId>,
}

impl Array {
    pub fn new(values: Vec<NodeId>) -> Self {
        Self { values }
    }
}

impl From<Array> for Node {
    fn from(array: Array) -> Self {
        Self::Array(array)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Edge {
    Value,
    Effect,
}
