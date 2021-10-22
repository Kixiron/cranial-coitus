use crate::{
    graph::{Node, NodeId, OutputPort, Rvsdg},
    ir::{
        Add, Assign, Block, Call, Const, Eq, Gamma, Instruction, Load, Neg, Not, Store, Theta,
        Value, VarId,
    },
};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
    time::Instant,
};

#[derive(Debug)]
pub struct IrBuilder {
    instructions: Vec<Instruction>,
    values: BTreeMap<OutputPort, Value>,
    evaluated: BTreeSet<NodeId>,
    evaluation_stack: Vec<NodeId>,
    top_level: bool,
}

impl IrBuilder {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            values: BTreeMap::new(),
            evaluated: BTreeSet::new(),
            evaluation_stack: Vec::new(),
            top_level: true,
        }
    }

    fn inst(&mut self, inst: impl Into<Instruction>) {
        self.instructions.push(inst.into());
    }

    pub fn translate(&mut self, graph: &Rvsdg) -> Block {
        let start_time = Instant::now();

        let old_level = self.top_level;
        self.top_level = false;

        self.evaluation_stack.extend(graph.node_ids());
        self.evaluation_stack.sort_unstable();

        while let Some(node) = self.evaluation_stack.pop() {
            if !self.evaluated.contains(&node) {
                self.push(graph, graph.get_node(node));
            }
        }

        let block = Block::new(mem::take(&mut self.instructions));

        self.top_level = old_level;

        if self.top_level {
            let elapsed = start_time.elapsed();
            tracing::debug!(
                target: "timings",
                "took {:#?} to sequentialize rvsdg",
                elapsed,
            );
        }

        block
    }

    pub fn push(&mut self, graph: &Rvsdg, node: &Node) {
        if self.evaluated.contains(&node.node_id()) {
            return;
        }

        let mut inputs: Vec<_> = graph
            .try_inputs(node.node_id())
            .flat_map(|(input, data)| data.map(|(node, output, kind)| (input, node, output, kind)))
            .collect();
        inputs.sort_unstable_by_key(|(_, input, _, _)| input.node_id());

        let mut input_values = BTreeMap::new();
        for (input, input_node, output, _) in inputs {
            if !self.evaluated.contains(&input_node.node_id()) {
                self.evaluation_stack.push(input_node.node_id());
                return;
            }

            if let Some(value) = self.values.get(&output).cloned() {
                input_values.insert(input, value);
            } else {
                tracing::error!(
                    "expected an input value for a value edge from {:?} (port {:?}) to {:?} (port {:?})",
                    input_node,
                    output,
                    node,
                    input,
                );
            }
        }

        match node {
            &Node::Int(int, value) => {
                let var = VarId::new(int.node().0);
                self.inst(crate::ir::Assign::new(var, Const::Int(value)));

                self.values.insert(int.value(), var.into());
            }

            &Node::Bool(bool, value) => {
                let var = VarId::new(bool.node().0);
                self.inst(Assign::new(var, Const::Bool(value)));

                self.values.insert(bool.value(), var.into());
            }

            Node::Add(add) => {
                let var = VarId::new(add.node().0);

                let lhs = input_values
                    .get(&add.lhs())
                    .cloned()
                    .unwrap_or(Value::Missing);
                let rhs = input_values
                    .get(&add.rhs())
                    .cloned()
                    .unwrap_or(Value::Missing);
                self.inst(Assign::new(var, Add::new(lhs, rhs)));

                self.values.insert(add.value(), var.into());
            }

            Node::Eq(eq) => {
                let var = VarId::new(eq.node().0);

                let lhs = input_values
                    .get(&eq.lhs())
                    .cloned()
                    .unwrap_or(Value::Missing);
                let rhs = input_values
                    .get(&eq.rhs())
                    .cloned()
                    .unwrap_or(Value::Missing);
                self.inst(Assign::new(var, Eq::new(lhs, rhs)));

                self.values.insert(eq.value(), var.into());
            }

            Node::Not(not) => {
                let var = VarId::new(not.node().0);

                let value = input_values
                    .get(&not.input())
                    .cloned()
                    .unwrap_or(Value::Missing);
                self.inst(Assign::new(var, Not::new(value)));

                self.values.insert(not.value(), var.into());
            }

            Node::Neg(neg) => {
                let var = VarId::new(neg.node().0);

                let value = input_values
                    .get(&neg.input())
                    .cloned()
                    .unwrap_or(Value::Missing);
                self.inst(Assign::new(var, Neg::new(value)));

                self.values.insert(neg.value(), var.into());
            }

            Node::Load(load) => {
                let var = VarId::new(load.node().0);

                let ptr = input_values
                    .get(&load.ptr())
                    .cloned()
                    .unwrap_or(Value::Missing);
                self.inst(Assign::new(
                    var,
                    Load::new(
                        ptr,
                        graph
                            .try_input(load.effect_in())
                            .and_then(|(_, output, _)| self.values.get(&output))
                            .and_then(Value::as_var),
                    ),
                ));

                self.values.insert(load.value(), var.into());
                self.values.insert(load.effect(), var.into());
            }

            Node::Store(store) => {
                let var = VarId::new(store.node().0);

                let ptr = input_values
                    .get(&store.ptr())
                    .cloned()
                    .unwrap_or(Value::Missing);
                let value = input_values
                    .get(&store.value())
                    .cloned()
                    .unwrap_or(Value::Missing);
                self.inst(Store::new(
                    ptr,
                    value,
                    graph
                        .try_input(store.effect_in())
                        .and_then(|(_, output, _)| self.values.get(&output))
                        .and_then(Value::as_var),
                ));

                self.values.insert(store.effect(), var.into());
            }

            Node::Input(input) => {
                let var = VarId::new(input.node().0);

                self.inst(Assign::new(
                    var,
                    Call::new(
                        "input",
                        Vec::new(),
                        graph
                            .try_input(input.effect_in())
                            .and_then(|(_, output, _)| self.values.get(&output))
                            .and_then(Value::as_var),
                    ),
                ));

                self.values.insert(input.value(), var.into());
                self.values.insert(input.effect(), var.into());
            }

            Node::Output(output) => {
                let var = VarId::new(output.node().0);

                let value = input_values
                    .get(&output.value())
                    .cloned()
                    .unwrap_or(Value::Missing);
                self.inst(Call::new(
                    "output",
                    vec![value],
                    graph
                        .try_input(output.effect_in())
                        .and_then(|(_, output, _)| self.values.get(&output))
                        .and_then(Value::as_var),
                ));

                self.values.insert(output.effect(), var.into());
            }

            Node::Theta(theta) => {
                let var = VarId::new(theta.node().0);
                let mut builder = Self {
                    values: BTreeMap::new(),
                    instructions: Vec::new(),
                    evaluated: BTreeSet::new(),
                    evaluation_stack: Vec::new(),
                    top_level: false,
                };

                for (input, &param) in theta.inputs().iter().zip(theta.input_params()) {
                    let port = theta.body().outputs(param).next().unwrap().0;
                    let value = input_values.get(input).cloned().unwrap_or(Value::Missing);

                    builder.values.insert(port, value);
                }

                let body = builder.translate(theta.body());

                for (&output, &param) in theta.outputs().iter().zip(theta.output_params()) {
                    let value = theta
                        .body()
                        .try_inputs(param)
                        .find_map(|(_, data)| data)
                        .map(|(_, output, _)| output)
                        .and_then(|output| builder.values.get(&output).cloned())
                        .unwrap_or(Value::Missing);

                    self.values.insert(output, value);
                }

                let cond = theta.body().get_node(theta.condition()).to_output_param();
                let cond = theta
                    .body()
                    .try_input(cond.value())
                    .map(|(_, port, _)| port)
                    .and_then(|cond| builder.values.get(&cond).cloned())
                    .unwrap_or(Value::Missing);

                if cond.is_missing() {
                    tracing::error!(
                        "failed to get condition for theta {:?} (cond: {:?})",
                        theta.node(),
                        theta.condition(),
                    );
                }

                self.inst(Theta::new(
                    body.into_inner(),
                    cond,
                    graph
                        .try_input(theta.effect_in())
                        .and_then(|(_, output, _)| self.values.get(&output))
                        .and_then(Value::as_var),
                ));

                self.values.insert(theta.effect_out(), var.into());
            }

            Node::Gamma(gamma) => {
                let var = VarId::new(gamma.node().0);
                let cond = input_values
                    .get(&gamma.condition())
                    .cloned()
                    .unwrap_or(Value::Missing);

                let mut truthy_builder = Self {
                    values: BTreeMap::new(),
                    instructions: Vec::new(),
                    evaluated: BTreeSet::new(),
                    evaluation_stack: Vec::new(),
                    top_level: false,
                };

                for (input, &[param, _]) in gamma.inputs().iter().zip(gamma.input_params()) {
                    let port = gamma.true_branch().outputs(param).next().unwrap().0;
                    let value = input_values.get(input).cloned().unwrap_or(Value::Missing);

                    truthy_builder.values.insert(port, value);
                }

                let truthy = truthy_builder.translate(gamma.true_branch());

                for (&output, &[param, _]) in gamma.outputs().iter().zip(gamma.output_params()) {
                    let value = truthy_builder
                        .values
                        .get(&gamma.true_branch().inputs(param).next().unwrap().2)
                        .cloned()
                        .unwrap_or(Value::Missing);

                    self.values.insert(output, value);
                }

                let mut falsy_builder = Self {
                    values: BTreeMap::new(),
                    instructions: Vec::new(),
                    evaluated: BTreeSet::new(),
                    evaluation_stack: Vec::new(),
                    top_level: false,
                };

                for (input, &[_, param]) in gamma.inputs().iter().zip(gamma.input_params()) {
                    let port = gamma.false_branch().outputs(param).next().unwrap().0;
                    let value = input_values.get(input).cloned().unwrap_or(Value::Missing);

                    falsy_builder.values.insert(port, value);
                }

                let falsy = falsy_builder.translate(gamma.false_branch());

                for (&output, &[_, param]) in gamma.outputs().iter().zip(gamma.output_params()) {
                    let value = gamma
                        .false_branch()
                        .try_inputs(param)
                        .find_map(|(_, data)| {
                            data.and_then(|(_, output, _)| falsy_builder.values.get(&output))
                        })
                        .cloned()
                        .unwrap_or(Value::Missing);

                    self.values.insert(output, value);
                }

                self.inst(Gamma::new(
                    cond,
                    truthy.into_inner(),
                    falsy.into_inner(),
                    graph
                        .try_input(gamma.effect_in())
                        .and_then(|(_, output, _)| self.values.get(&output))
                        .and_then(Value::as_var),
                ));

                self.values.insert(gamma.effect_out(), var.into());
            }

            Node::Start(start) => {
                let var = VarId::new(start.node().0);
                self.values.insert(start.effect(), var.into());
            }

            Node::InputPort(_) | Node::OutputPort(_) | Node::End(_) => {}
        };

        self.evaluated.insert(node.node_id());

        let mut outputs: Vec<_> = graph
            .outputs(node.node_id())
            .flat_map(|(_, data)| data.map(|(node, _, _)| node.node_id()))
            .collect();
        outputs.sort_unstable();
        self.evaluation_stack.extend(outputs);
    }
}
