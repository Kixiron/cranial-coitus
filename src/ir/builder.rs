use crate::{
    graph::{EdgeKind, Node, NodeId, OutputPort, Rvsdg},
    ir::{
        Add, Assign, Block, Call, Const, Eq, Instruction, Load, Not, Phi, Store, Theta, Value,
        VarId,
    },
};
use std::{
    collections::{HashMap, HashSet},
    mem,
};

#[derive(Debug)]
pub struct IrBuilder {
    instructions: Vec<Instruction>,
    values: HashMap<OutputPort, Value>,
    evaluated: HashSet<NodeId>,
    evaluation_stack: Vec<NodeId>,
}

impl IrBuilder {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            values: HashMap::new(),
            evaluated: HashSet::new(),
            evaluation_stack: Vec::new(),
        }
    }

    fn inst(&mut self, inst: impl Into<Instruction>) {
        self.instructions.push(inst.into());
    }

    pub fn translate(&mut self, graph: &Rvsdg) -> Block {
        self.evaluation_stack.extend(graph.nodes());
        self.evaluation_stack.sort_unstable();

        while let Some(node) = self.evaluation_stack.pop() {
            if !self.evaluated.contains(&node) {
                self.push(graph, graph.get_node(node));
            }
        }

        Block::new(mem::take(&mut self.instructions))
    }

    pub fn push(&mut self, graph: &Rvsdg, node: &Node) {
        if self.evaluated.contains(&node.node_id()) {
            return;
        }

        let mut inputs: Vec<_> = graph.inputs(node.node_id()).collect();
        inputs.sort_unstable_by_key(|(_, input, _, _)| input.node_id());

        let mut input_values = HashMap::new();
        for (input, input_node, output, kind) in inputs {
            if !self.evaluated.contains(&input_node.node_id()) {
                self.evaluation_stack.push(input_node.node_id());
                return;
            }

            if let Some(value) = self.values.get(&output).cloned() {
                input_values.insert(input, value);
            } else {
                debug_assert_eq!(
                    kind,
                    EdgeKind::Effect,
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

                let lhs = input_values[&add.lhs()].clone();
                let rhs = input_values[&add.rhs()].clone();
                self.inst(Assign::new(var, Add::new(lhs, rhs)));

                self.values.insert(add.value(), var.into());
            }

            Node::Eq(eq) => {
                let var = VarId::new(eq.node().0);

                let lhs = input_values[&eq.lhs()].clone();
                let rhs = input_values[&eq.rhs()].clone();
                self.inst(Assign::new(var, Eq::new(lhs, rhs)));

                self.values.insert(eq.value(), var.into());
            }

            Node::Not(not) => {
                let var = VarId::new(not.node().0);

                let value = input_values[&not.input()].clone();
                self.inst(Assign::new(var, Not::new(value)));

                self.values.insert(not.value(), var.into());
            }

            Node::Load(load) => {
                let var = VarId::new(load.node().0);

                let ptr = input_values[&load.ptr()].clone();
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

                let ptr = input_values[&store.ptr()].clone();
                let value = input_values[&store.value()].clone();
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

                let value = input_values[&output.value()].clone();
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
                    values: HashMap::new(),
                    instructions: Vec::new(),
                    evaluated: HashSet::new(),
                    evaluation_stack: Vec::new(),
                };

                for (input, &param) in theta.inputs().iter().zip(theta.input_params()) {
                    let port = theta.body().outputs(param).next().unwrap().0;
                    let value = input_values[input].clone();

                    builder.values.insert(port, value);
                }

                let body = builder.translate(theta.body());

                for (&output, &param) in theta.outputs().iter().zip(theta.output_params()) {
                    let value =
                        builder.values[&theta.body().inputs(param).next().unwrap().2].clone();

                    self.values.insert(output, value);
                }

                let cond = theta.body().get_node(theta.condition()).to_output_param();
                let cond = builder
                    .values
                    .get(&theta.body().get_input(cond.value()).1)
                    .cloned();

                if cond.is_none() {
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

            Node::Phi(phi) => {
                let var = VarId::new(phi.node().0);
                let cond = input_values[&phi.condition()].clone();

                let mut truthy_builder = Self {
                    values: HashMap::new(),
                    instructions: Vec::new(),
                    evaluated: HashSet::new(),
                    evaluation_stack: Vec::new(),
                };

                for (input, &[param, _]) in phi.inputs().iter().zip(phi.input_params()) {
                    let port = phi.truthy().outputs(param).next().unwrap().0;
                    let value = input_values[input].clone();

                    truthy_builder.values.insert(port, value);
                }

                let truthy = truthy_builder.translate(phi.truthy());

                for (&output, &[param, _]) in phi.outputs().iter().zip(phi.output_params()) {
                    let value = truthy_builder.values
                        [&phi.truthy().inputs(param).next().unwrap().2]
                        .clone();

                    self.values.insert(output, value);
                }

                let mut falsy_builder = Self {
                    values: HashMap::new(),
                    instructions: Vec::new(),
                    evaluated: HashSet::new(),
                    evaluation_stack: Vec::new(),
                };

                for (input, &[_, param]) in phi.inputs().iter().zip(phi.input_params()) {
                    let port = phi.falsy().outputs(param).next().unwrap().0;
                    let value = input_values[input].clone();

                    falsy_builder.values.insert(port, value);
                }

                let falsy = falsy_builder.translate(phi.falsy());

                for (&output, &[_, param]) in phi.outputs().iter().zip(phi.output_params()) {
                    let value =
                        falsy_builder.values[&phi.falsy().inputs(param).next().unwrap().2].clone();

                    self.values.insert(output, value);
                }

                self.inst(Phi::new(
                    cond,
                    truthy.into_inner(),
                    falsy.into_inner(),
                    graph
                        .try_input(phi.effect_in())
                        .and_then(|(_, output, _)| self.values.get(&output))
                        .and_then(Value::as_var),
                ));

                self.values.insert(phi.effect_out(), var.into());
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
