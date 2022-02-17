use crate::{
    jit::{
        basic_block::{BasicBlock, BlockId, Blocks, Branch, Instruction, Terminator, ValId, Value},
        block_visitor::BasicBlockVisitor,
    },
    utils::AssertNone,
};
use petgraph::{
    algo::dominators::{self, Dominators},
    graph::NodeIndex,
    Graph,
};
use std::{
    cmp::{max, min, Ordering},
    collections::BTreeMap,
    fmt::{self, Debug, Display},
};

#[derive(Debug)]
pub struct Liveliness {
    pub(super) lifetimes: BTreeMap<ValId, Lifetime>,
}

impl Liveliness {
    pub fn new() -> Self {
        Self {
            lifetimes: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Lifetime {
    pub(super) blocks: Vec<(BlockId, BlockLifetime)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BlockLifetime {
    Whole,
    Span(Location, Location),
}

impl BlockLifetime {
    pub const fn is_whole(&self) -> bool {
        matches!(self, Self::Whole)
    }
}

impl Debug for BlockLifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Whole => write!(f, "Whole"),
            Self::Span(start, finish) => f.debug_tuple("Span").field(&(start..finish)).finish(),
        }
    }
}

pub struct LivelinessAnalysis {
    first_usages: BTreeMap<ValId, Location>,
    last_usages: BTreeMap<ValId, Location>,
    usages: BTreeMap<ValId, Vec<Location>>,
    declarations: BTreeMap<ValId, Location>,
    current_block: BlockId,
    current_inst: Option<usize>,
}

impl LivelinessAnalysis {
    pub fn new() -> Self {
        Self {
            first_usages: BTreeMap::new(),
            last_usages: BTreeMap::new(),
            usages: BTreeMap::new(),
            declarations: BTreeMap::new(),
            current_block: BlockId::new(u32::MAX),
            current_inst: None,
        }
    }

    pub fn run(&mut self, blocks: &Blocks) -> Liveliness {
        self.visit_blocks(blocks);

        // Sort all usages
        for usages in self.usages.values_mut() {
            usages.sort_unstable();
        }

        let graph = BlockGraphBuilder::new().build(blocks);

        let mut lifetimes: BTreeMap<ValId, Lifetime> = BTreeMap::new();

        let mut dominators = Vec::new();
        for (value, usages) in &self.usages {
            dominators.clear();
            dominators.extend(
                usages
                    .iter()
                    // Filter out the declaration
                    .filter(|&&usage| usage != self.declarations[value])
                    // Find all dominators
                    .filter_map(|usage| {
                        graph
                            .dominators
                            .dominators(graph.nodes[&usage.block()])
                            .map(|dominators| (usage, dominators))
                    })
                    .flat_map(|(&usage, dominators)| {
                        dominators.map(move |dominator| (usage, dominator))
                    })
                    .map(|(usage, dominator)| (usage, graph.graph[dominator])),
            );

            let declaration = self.declarations[value];

            for (location, dominator) in dominators.iter().filter(|&&(location, dominator)| {
                location < declaration && dominator >= declaration.block()
            }) {
                println!(
                    "{} is within a loop (used at {}, dominated by {})",
                    value, location, dominator,
                );

                lifetimes
                    .entry(*value)
                    .or_default()
                    .blocks
                    .push((location.block(), BlockLifetime::Whole));
            }
        }

        // Add all lifetime spans
        // for (&value, &start) in &self.first_usages {
        //     let end = self.last_usages[&value];
        //
        //     lifetimes
        //         .entry(value)
        //         .or_default()
        //         .blocks
        //         .push((start.block(), BlockLifetime::Span(start, end)));
        // }
        for (&value, usages) in &self.usages {
            for usage in usages {
                lifetimes
                    .entry(value)
                    .or_default()
                    .blocks
                    .push((usage.block(), BlockLifetime::Whole));
            }
        }

        // Sort and process the added lifetimes
        for lifetime in lifetimes.values_mut() {
            lifetime.blocks.sort_by_key(|&(block, _)| block);
            lifetime.blocks.dedup();

            let mut idx = 0;
            while idx + 1 < lifetime.blocks.len() {
                let (block_one, lifetime_one) = &lifetime.blocks[idx];
                let (block_two, lifetime_two) = &lifetime.blocks[idx + 1];

                if block_one == block_two {
                    if lifetime_one.is_whole() {
                        lifetime.blocks.remove(idx + 1);
                    } else if lifetime_two.is_whole() {
                        lifetime.blocks.remove(idx);
                        continue;
                    }
                }

                idx += 1;
            }
        }

        println!("lifetimes: {:#?}", lifetimes);

        Liveliness { lifetimes }
    }

    /// Returns `true` if the given value is alive at the given location
    pub fn is_alive(&self, value: ValId, location: Location) -> bool {
        location >= *self.first_usages.get(&value).unwrap()
            && location <= *self.last_usages.get(&value).unwrap()
    }

    /// Returns all values declared at the current location
    pub fn declarations_at(&self, location: Location) -> Vec<ValId> {
        let mut declarations = Vec::new();
        self.declarations_at_into(location, &mut declarations);
        declarations
    }

    /// Returns all values declared at the current location
    pub fn declarations_at_into(&self, location: Location, declarations: &mut Vec<ValId>) {
        declarations.extend(
            self.first_usages
                .iter()
                .filter_map(|(&value, usage)| location.eq(usage).then_some(value)),
        );
    }

    /// Returns all values which are alive at the current location
    pub fn live_values(&self, location: Location) -> Vec<ValId> {
        let mut live_values = Vec::new();
        self.live_values_into(location, &mut live_values);
        live_values
    }

    /// Returns all values which are alive at the current location
    pub fn live_values_into(&self, location: Location, live_values: &mut Vec<ValId>) {
        live_values.extend(self.first_usages.iter().filter_map(|(&value, usage)| {
            location.ge(usage).then_some(value).and_then(|value| {
                let usage = self.last_usages.get(&value).unwrap();
                location.le(usage).then_some(value)
            })
        }));
    }

    /// Returns `true` if the given instruction of the given block is the *first* usage of the given [`ValId`].
    /// A [`None`] given for `index` means a block terminator instead of an instruction in the block's body
    #[allow(dead_code)]
    pub fn is_first_usage(&self, value: ValId, location: Location) -> bool {
        self.first_usages.get(&value).copied() == Some(location)
    }

    /// Returns `true` if the given instruction of the given block is the *last* usage of the given [`ValId`].
    /// A [`None`] given for `index` means a block terminator instead of an instruction in the block's body
    pub fn is_last_usage(&self, value: ValId, location: Location) -> bool {
        self.last_usages.get(&value).copied() == Some(location)
    }

    /// Returns `true` if the given instruction of the given block is the *only* usage of the given [`ValId`].
    /// A [`None`] given for `index` means a block terminator instead of an instruction in the block's body
    #[allow(dead_code)]
    pub fn is_only_usage(&self, value: ValId, location: Location) -> bool {
        // If an instruction is both the first and last usage of a value then it should be the only user.
        // This doesn't account for instructions that use a value twice, like `add %x, %x`, but that shouldn't
        // be a problem
        self.is_first_usage(value, location) && self.is_last_usage(value, location)
    }

    fn add_usage(&mut self, value: ValId) {
        let location = Location::new(self.current_block, self.current_inst);

        // Denote the usage
        self.usages
            .entry(value)
            .or_insert_with(Vec::new)
            .push(location);

        // Use the minimal location as the first usage
        self.first_usages
            .entry(value)
            .and_modify(|usage| *usage = min(*usage, location))
            .or_insert(location);

        // Use the maximal location as the final usage
        self.last_usages
            .entry(value)
            .and_modify(|usage| *usage = max(*usage, location))
            .or_insert(location);
    }
}

impl BasicBlockVisitor for LivelinessAnalysis {
    fn visit_value(&mut self, value: Value) {
        if let Value::Val(value) = value {
            self.add_usage(value);
        }
    }

    fn before_visit_block(&mut self, block: &BasicBlock) {
        self.current_block = block.id();
    }

    fn before_visit_inst(&mut self, inst: &Instruction) {
        match &mut self.current_inst {
            Some(idx) => *idx += 1,
            current @ None => *current = Some(0),
        }

        if let Instruction::Assign(assign) = inst {
            self.add_usage(assign.value());
            self.declarations
                .insert(
                    assign.value(),
                    Location::new(self.current_block, self.current_inst),
                )
                .debug_unwrap_none();
        }
    }

    fn before_visit_term(&mut self, _term: &Terminator) {
        self.current_inst = None;
    }
}

impl Debug for LivelinessAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.first_usages.iter().map(|(&val, &start)| {
                let end = *self.last_usages.get(&val).unwrap();
                (val.display(), start..end)
            }))
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Location {
    block: BlockId,
    inst: Option<usize>,
}

impl Location {
    pub fn new(block: BlockId, inst: Option<usize>) -> Self {
        Self { block, inst }
    }

    pub fn block(&self) -> BlockId {
        self.block
    }

    pub fn block_mut(&mut self) -> &mut BlockId {
        &mut self.block
    }

    pub fn inst(&self) -> Option<usize> {
        self.inst
    }

    pub fn inst_mut(&mut self) -> &mut Option<usize> {
        &mut self.inst
    }
}

impl PartialOrd for Location {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Location {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.block.cmp(&other.block) {
            Ordering::Equal => {}
            ord => return ord,
        }

        match (self.inst, other.inst) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(inst1), Some(inst2)) => inst1.cmp(&inst2),
        }
    }
}

impl Debug for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.inst {
            Some(idx) => write!(f, "{}+{}", self.block, idx),
            None => write!(f, "{}+term", self.block),
        }
    }
}

impl Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum JumpTarget {
    Diverge,
    Block(BlockId),
}

impl Default for JumpTarget {
    fn default() -> Self {
        Self::Diverge
    }
}

#[derive(Debug)]
struct BlockGraph {
    graph: Graph<BlockId, ()>,
    dominators: Dominators<NodeIndex>,
    nodes: BTreeMap<BlockId, NodeIndex>,
}

#[derive(Debug)]
struct BlockGraphBuilder {
    graph: Graph<BlockId, ()>,
    nodes: BTreeMap<BlockId, NodeIndex>,
    current: NodeIndex,
}

impl BlockGraphBuilder {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            nodes: BTreeMap::new(),
            current: NodeIndex::new(0),
        }
    }

    pub fn build(mut self, blocks: &Blocks) -> BlockGraph {
        self.graph.reserve_nodes(blocks.len());
        self.graph.reserve_edges(blocks.len() * 2);

        for block in blocks {
            let node = self.graph.add_node(block.id());
            self.nodes.insert(block.id(), node).debug_unwrap_none();
        }

        self.visit_blocks(blocks);
        self.graph.shrink_to_fit();

        let dominators = dominators::simple_fast(&self.graph, self.nodes[&blocks[0].id()]);
        BlockGraph {
            graph: self.graph,
            dominators,
            nodes: self.nodes,
        }
    }
}

impl BasicBlockVisitor for BlockGraphBuilder {
    fn before_visit_block(&mut self, block: &BasicBlock) {
        self.current = self.nodes[&block.id()];
    }

    fn visit_jump(&mut self, target: &BlockId) {
        let target = self.nodes[target];
        self.graph.add_edge(self.current, target, ());
    }

    fn visit_branch(&mut self, branch: &Branch) {
        let true_jump = self.nodes[&branch.true_jump()];
        self.graph.add_edge(self.current, true_jump, ());

        let false_jump = self.nodes[&branch.false_jump()];
        self.graph.add_edge(self.current, false_jump, ());
    }
}
