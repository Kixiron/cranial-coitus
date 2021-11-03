use crate::{
    graph::{InputParam, Node, NodeExt, NodeId, OutputParam, OutputPort, Rvsdg},
    ir::{
        Add, Assign, Block, Call, Const, EffectId, Eq, Gamma, Instruction, Load, Neg, Not, Store,
        Theta, Value, VarId, Variance,
    },
    utils::AssertNone,
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
        let mut block = Block::new();
        self.translate_into(&mut block, graph);
        block
    }

    pub fn translate_into(&mut self, block: &mut Block, graph: &Rvsdg) {
        let start_time = Instant::now();

        debug_assert!(self.instructions.is_empty());
        mem::swap(&mut block.instructions, &mut self.instructions);

        let old_level = self.top_level;
        self.top_level = false;

        self.evaluation_stack.extend(graph.node_ids());
        self.evaluation_stack.sort_unstable();

        while let Some(node) = self.evaluation_stack.pop() {
            if !self.evaluated.contains(&node) {
                self.push(graph, graph.get_node(node));
            }
        }

        self.top_level = old_level;

        if self.top_level {
            let elapsed = start_time.elapsed();
            tracing::debug!(
                target: "timings",
                "took {:#?} to sequentialize rvsdg",
                elapsed,
            );
        }

        mem::swap(&mut block.instructions, &mut self.instructions);
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
        for &(input, input_node, output, _) in &inputs {
            if !self.evaluated.contains(&input_node.node_id()) {
                self.evaluation_stack.push(input_node.node_id());
                self.evaluation_stack
                    .extend(inputs.iter().map(|(_, node, ..)| node.node_id()));

                return;
            }

            if let Some(value) = self.values.get(&output).cloned() {
                input_values.insert(input, value);
            } else {
                // tracing::error!(
                //     "expected an input value for a value edge from {:?} (port {:?}) to {:?} (port {:?})",
                //     input_node,
                //     output,
                //     node,
                //     input,
                // );
            }
        }

        match node {
            &Node::Int(int, value) => {
                let var = VarId::new(int.value());
                self.inst(crate::ir::Assign::new(var, Const::Int(value)));

                self.values
                    // .insert(int.value(), Const::Int(value).into())
                    .insert(int.value(), var.into())
                    .debug_unwrap_none();
            }

            &Node::Bool(bool, value) => {
                let var = VarId::new(bool.value());
                self.inst(Assign::new(var, Const::Bool(value)));

                self.values
                    // .insert(bool.value(), Const::Bool(value).into())
                    .insert(bool.value(), var.into())
                    .debug_unwrap_none();
            }

            Node::Add(add) => {
                let var = VarId::new(add.value());

                let lhs = input_values
                    .get(&add.lhs())
                    .cloned()
                    .unwrap_or(Value::Missing);
                let rhs = input_values
                    .get(&add.rhs())
                    .cloned()
                    .unwrap_or(Value::Missing);
                self.inst(Assign::new(var, Add::new(lhs, rhs)));

                self.values
                    .insert(add.value(), var.into())
                    .debug_unwrap_none();
            }

            Node::Eq(eq) => {
                let var = VarId::new(eq.value());

                let lhs = input_values
                    .get(&eq.lhs())
                    .cloned()
                    .unwrap_or(Value::Missing);
                let rhs = input_values
                    .get(&eq.rhs())
                    .cloned()
                    .unwrap_or(Value::Missing);
                self.inst(Assign::new(var, Eq::new(lhs, rhs)));

                self.values
                    .insert(eq.value(), var.into())
                    .debug_unwrap_none();
            }

            Node::Not(not) => {
                let var = VarId::new(not.value());

                let value = input_values
                    .get(&not.input())
                    .cloned()
                    .unwrap_or(Value::Missing);
                self.inst(Assign::new(var, Not::new(value)));

                self.values
                    .insert(not.value(), var.into())
                    .debug_unwrap_none();
            }

            Node::Neg(neg) => {
                let var = VarId::new(neg.value());

                let value = input_values
                    .get(&neg.input())
                    .cloned()
                    .unwrap_or(Value::Missing);
                self.inst(Assign::new(var, Neg::new(value)));

                self.values
                    .insert(neg.value(), var.into())
                    .debug_unwrap_none();
            }

            Node::Load(load) => {
                let value = VarId::new(load.value());
                let effect = EffectId::new(load.effect());

                let ptr = input_values
                    .get(&load.ptr())
                    .cloned()
                    .unwrap_or(Value::Missing);
                self.inst(Assign::new(
                    value,
                    Load::new(
                        ptr,
                        effect,
                        graph
                            .try_input(load.effect_in())
                            .map(|(_, output, _)| EffectId::new(output)),
                    ),
                ));

                self.values
                    .insert(load.value(), value.into())
                    .debug_unwrap_none();
            }

            Node::Store(store) => {
                let effect = EffectId::new(store.effect());

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
                    effect,
                    graph
                        .try_input(store.effect_in())
                        .map(|(_, output, _)| EffectId::new(output)),
                ));
            }

            Node::Input(input) => {
                let value = VarId::new(input.value());
                let effect = EffectId::new(input.effect());

                let call = Call::new(
                    input.node(),
                    "input",
                    Vec::new(),
                    effect,
                    graph
                        .try_input(input.effect_in())
                        .map(|(_, output, _)| EffectId::new(output)),
                );

                self.inst(Assign::new(value, call));

                self.values
                    .insert(input.value(), value.into())
                    .debug_unwrap_none();
            }

            Node::Output(output) => {
                let effect = EffectId::new(output.effect());

                let value = input_values
                    .get(&output.value())
                    .cloned()
                    .unwrap_or(Value::Missing);

                let call = Call::new(
                    output.node(),
                    "output",
                    vec![value],
                    effect,
                    graph
                        .try_input(output.effect_in())
                        .map(|(_, output, _)| EffectId::new(output)),
                );

                self.inst(call);
            }

            Node::Theta(theta) => {
                let mut builder = Self {
                    values: BTreeMap::new(),
                    instructions: Vec::new(),
                    evaluated: BTreeSet::new(),
                    evaluation_stack: Vec::new(),
                    top_level: false,
                };
                let (mut body, mut inputs, mut input_vars) = (
                    Block::with_capacity(theta.inputs_len() + theta.outputs_len()),
                    BTreeMap::new(),
                    BTreeMap::new(),
                );

                // Create all the invariant input params for the theta body
                for (input, param) in theta.invariant_input_pairs() {
                    let input_id = VarId::new(param.output());
                    let value = input_values.get(&input).cloned().unwrap_or(Value::Missing);
                    body.push(Assign::input(input_id, value.clone(), Variance::Invariant).into());

                    if value.is_missing() {
                        tracing::warn!(
                            "missing invariant input value for theta body input {:?}: {:?}",
                            input,
                            param,
                        );
                    }

                    builder
                        .values
                        .insert(param.output(), input_id.into())
                        .debug_unwrap_none();
                }

                // Create all the variant input params for the theta body
                for (input, param, feedback) in theta.variant_input_pair_ids_with_feedback() {
                    let param = theta.body().to_node::<InputParam>(param);
                    let input_id = VarId::new(param.output());
                    let value = input_values.get(&input).cloned().unwrap_or(Value::Missing);
                    let feedback_from = VarId::new(feedback);

                    body.push(
                        Assign::input(input_id, value.clone(), Variance::Variant { feedback_from })
                            .into(),
                    );

                    if value.is_missing() {
                        tracing::warn!(
                            "missing variant input value for theta body input {:?}: {:?}",
                            input,
                            param,
                        );
                    }

                    inputs.insert(input_id, value.clone()).debug_unwrap_none();
                    input_vars.insert(input, input_id).debug_unwrap_none();

                    builder
                        .values
                        .insert(param.output(), input_id.into())
                        .debug_unwrap_none();
                }

                builder.translate_into(&mut body, theta.body());
                let (mut outputs, mut output_feedback) = (BTreeMap::new(), BTreeMap::new());
                for (output, param) in theta.output_pairs() {
                    let value = theta
                        .body()
                        .try_input_source(param.input())
                        .and_then(|output| builder.values.get(&output).cloned())
                        .unwrap_or(Value::Missing);

                    if value.is_missing() {
                        tracing::warn!(
                            "missing output value for theta body {:?}: {:?}",
                            output,
                            param,
                        );
                    }

                    let output_id = VarId::new(output);
                    body.push(Instruction::Assign(Assign::output(
                        output_id,
                        value.clone(),
                    )));

                    self.values
                        .insert(output, output_id.into())
                        .debug_unwrap_none();

                    outputs.insert(output_id, value).debug_unwrap_none();
                    output_feedback
                        .insert(output_id, input_vars[&theta.output_feedback()[&output]])
                        .debug_unwrap_none();
                }

                let condition_node = theta.body().cast_node::<OutputParam>(theta.condition_id());
                let condition = condition_node
                    .and_then(|cond| theta.body().try_input_source(cond.input()))
                    .and_then(|cond| builder.values.get(&cond).cloned())
                    .unwrap_or(Value::Missing);
                self.values.extend(builder.values);

                if condition.is_missing() {
                    tracing::error!(
                        "failed to get condition for theta {:?} (cond: {:?})",
                        theta.node(),
                        condition_node,
                    );
                }

                let input_effect = theta
                    .input_effect()
                    .and_then(|effect| graph.try_input_source(effect).map(EffectId::new));

                self.inst(Theta::new(
                    theta.node(),
                    body.into_inner(),
                    condition,
                    theta.output_effect().map(EffectId::new),
                    input_effect,
                    inputs,
                    outputs,
                    output_feedback,
                ));
            }

            Node::Gamma(gamma) => {
                let cond = input_values
                    .get(&gamma.condition())
                    .cloned()
                    .unwrap_or(Value::Missing);

                if cond.is_missing() {
                    tracing::warn!(
                        "missing condition gamma node {:?}: {:?}",
                        gamma.node(),
                        gamma.condition(),
                    );
                }

                let mut truthy_builder = Self {
                    values: BTreeMap::new(),
                    instructions: Vec::new(),
                    evaluated: BTreeSet::new(),
                    evaluation_stack: Vec::new(),
                    top_level: false,
                };

                let mut truthy = Block::new();
                for (input, &[param, _]) in gamma.inputs().iter().zip(gamma.input_params()) {
                    let input_param = gamma.true_branch().to_node::<InputParam>(param);
                    let input_id = VarId::new(input_param.output());
                    let value = input_values.get(input).cloned().unwrap_or(Value::Missing);
                    truthy.push(Assign::input(input_id, value.clone(), Variance::None).into());

                    if value.is_missing() {
                        tracing::warn!(
                            "missing input value for gamma true branch input {:?}: {:?}",
                            input,
                            input_param,
                        );
                    }

                    truthy_builder
                        .values
                        .insert(input_param.output(), value)
                        .debug_unwrap_none();
                }

                truthy_builder.translate_into(&mut truthy, gamma.true_branch());
                let mut true_outputs = BTreeMap::new();

                for (&output, &[param, _]) in gamma.outputs().iter().zip(gamma.output_params()) {
                    let output_param = gamma.true_branch().to_node::<OutputParam>(param);
                    let param_source = gamma.true_branch().get_input(output_param.input()).1;
                    let value = truthy_builder
                        .values
                        .get(&param_source)
                        .cloned()
                        .unwrap_or(Value::Missing);

                    if value.is_missing() {
                        tracing::warn!(
                            "missing output value for gamma true branch {:?}: {:?}",
                            output,
                            output_param,
                        );
                    }

                    let output_id = VarId::new(output);
                    truthy.push(Instruction::Assign(Assign::output(
                        output_id,
                        value.clone(),
                    )));

                    self.values
                        .insert(output, output_id.into())
                        .debug_unwrap_none();

                    true_outputs.insert(output_id, value).debug_unwrap_none();
                }
                self.values.extend(truthy_builder.values);

                let mut falsy_builder = Self {
                    values: BTreeMap::new(),
                    instructions: Vec::new(),
                    evaluated: BTreeSet::new(),
                    evaluation_stack: Vec::new(),
                    top_level: false,
                };

                let mut falsy = Block::new();
                for (input, &[_, param]) in gamma.inputs().iter().zip(gamma.input_params()) {
                    let input_param = gamma.false_branch().to_node::<InputParam>(param);
                    let input_id = VarId::new(input_param.output());
                    let value = input_values.get(input).cloned().unwrap_or(Value::Missing);
                    falsy.push(Assign::input(input_id, value.clone(), Variance::None).into());

                    if value.is_missing() {
                        tracing::warn!(
                            "missing input value for gamma false branch input {:?}: {:?}",
                            input,
                            input_param,
                        );
                    }

                    falsy_builder
                        .values
                        .insert(input_param.output(), value)
                        .debug_unwrap_none();
                }

                falsy_builder.translate_into(&mut falsy, gamma.false_branch());
                let mut false_outputs = BTreeMap::new();

                for (&output, &[_, param]) in gamma.outputs().iter().zip(gamma.output_params()) {
                    let output_param = gamma.false_branch().to_node::<OutputParam>(param);
                    let param_source = gamma.false_branch().get_input(output_param.input()).1;
                    let value = falsy_builder
                        .values
                        .get(&param_source)
                        .cloned()
                        .unwrap_or(Value::Missing);

                    if value.is_missing() {
                        tracing::warn!(
                            "missing output value for gamma false branch {:?}: {:?}",
                            output,
                            gamma.false_branch().get_node(param),
                        );
                    }

                    let output_id = VarId::new(output);
                    falsy.push(Instruction::Assign(Assign::output(
                        output_id,
                        value.clone(),
                    )));

                    // TODO: Still double-insert on this, need a merge node
                    self.values.insert(output, output_id.into());

                    false_outputs.insert(output_id, value).debug_unwrap_none();
                }
                self.values.extend(falsy_builder.values);

                let prev_effect = graph
                    .try_input(gamma.effect_in())
                    .map(|(_, output, _)| EffectId::new(output));

                if prev_effect.is_none() {
                    tracing::warn!(
                        "failed to get previous effect {:?} for gamma node {:?}",
                        gamma.effect_in(),
                        gamma.node(),
                    );
                }

                self.inst(Gamma::new(
                    gamma.node(),
                    cond,
                    truthy.into_inner(),
                    true_outputs,
                    falsy.into_inner(),
                    false_outputs,
                    EffectId::new(gamma.effect_out()),
                    prev_effect,
                ));
            }

            Node::InputParam(_) | Node::OutputParam(_) | Node::Start(_) | Node::End(_) => {}
        };

        self.evaluated.insert(node.node_id());

        let mut outputs: Vec<_> = node
            .outputs()
            .into_iter()
            .flat_map(|output| graph.get_outputs(output))
            .map(|(node, ..)| node.node_id())
            .collect();
        outputs.sort_unstable();
        self.evaluation_stack.extend(outputs);
    }
}
