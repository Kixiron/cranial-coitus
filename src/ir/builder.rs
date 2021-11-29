use crate::{
    graph::{self, InputParam, InputPort, Node, NodeExt, NodeId, OutputParam, OutputPort, Rvsdg},
    ir::{
        lifetime, Add, Assign, AssignTag, Block, Call, Const, EffectId, Eq, Expr, Gamma,
        Instruction, Load, Mul, Neg, Not, Store, Theta, Value, VarId, Variance,
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
    pub values: BTreeMap<OutputPort, Value>,
    evaluated: BTreeSet<NodeId>,
    evaluation_stack: Vec<NodeId>,
    top_level: bool,
    inline_constants: bool,
}

impl IrBuilder {
    pub fn new(inline_constants: bool) -> Self {
        Self {
            instructions: Vec::new(),
            values: BTreeMap::new(),
            evaluated: BTreeSet::new(),
            evaluation_stack: Vec::new(),
            top_level: true,
            inline_constants,
        }
    }

    pub fn finish(&mut self) -> Block {
        Block {
            instructions: mem::take(&mut self.instructions),
        }
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

            tracing::debug!(target: "timings", "started lifetime analysis");
            let start_time = Instant::now();

            lifetime::analyze_block(&mut self.instructions, |_| false);

            let elapsed = start_time.elapsed();
            tracing::debug!(
                target: "timings",
                "finished lifetime analysis in {:#?}",
                elapsed,
            );
        }

        mem::swap(&mut block.instructions, &mut self.instructions);
    }

    fn inst(&mut self, inst: impl Into<Instruction>) {
        self.instructions.push(inst.into());
    }

    pub fn push(&mut self, graph: &Rvsdg, node: &Node) {
        if self.evaluated.contains(&node.node()) {
            return;
        }

        let mut inputs: Vec<_> = graph
            .try_inputs(node.node())
            .flat_map(|(input, data)| data.map(|(node, output, kind)| (input, node, output, kind)))
            .collect();
        inputs.sort_unstable_by_key(|(_, input, _, _)| input.node());

        let mut input_values = BTreeMap::new();
        for &(input, input_node, output, _) in &inputs {
            if !self.evaluated.contains(&input_node.node()) {
                self.evaluation_stack.push(input_node.node());
                self.evaluation_stack
                    .extend(inputs.iter().map(|(_, node, ..)| node.node()));

                return;
            }

            if let Some(value) = self.values.get(&output).cloned() {
                input_values.insert(input, value);
            }
        }

        match node {
            &Node::Int(int, value) => {
                let var = VarId::new(int.value());
                self.inst(crate::ir::Assign::new(var, Const::Int(value)));

                let value = if self.inline_constants {
                    Const::Int(value).into()
                } else {
                    var.into()
                };
                self.values.insert(int.value(), value).debug_unwrap_none();
            }

            &Node::Bool(bool, value) => {
                let var = VarId::new(bool.value());
                self.inst(Assign::new(var, Const::Bool(value)));

                let value = if self.inline_constants {
                    Const::Bool(value).into()
                } else {
                    var.into()
                };
                self.values.insert(bool.value(), value).debug_unwrap_none();
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

            Node::Mul(mul) => {
                let var = VarId::new(mul.value());

                let lhs = input_values
                    .get(&mul.lhs())
                    .cloned()
                    .unwrap_or(Value::Missing);
                let rhs = input_values
                    .get(&mul.rhs())
                    .cloned()
                    .unwrap_or(Value::Missing);
                self.inst(Assign::new(var, Mul::new(lhs, rhs)));

                self.values
                    .insert(mul.value(), var.into())
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
                let value = VarId::new(load.output_value());
                let effect = EffectId::new(load.output_effect());

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
                            .try_input(load.input_effect())
                            .map(|(_, output, _)| EffectId::new(output)),
                    ),
                ));

                self.values
                    .insert(load.output_value(), value.into())
                    .debug_unwrap_none();
            }

            Node::Store(store) => {
                let effect = EffectId::new(store.output_effect());

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
                let value = VarId::new(input.output_value());
                let effect = EffectId::new(input.output_effect());

                let call = Call::new(
                    input.node(),
                    "input",
                    Vec::new(),
                    effect,
                    graph
                        .try_input(input.input_effect())
                        .map(|(_, output, _)| EffectId::new(output)),
                );

                self.inst(Assign::new(value, call));

                self.values
                    .insert(input.output_value(), value.into())
                    .debug_unwrap_none();
            }

            Node::Output(output) => {
                let effect = EffectId::new(output.output_effect());

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
                        .try_input(output.input_effect())
                        .map(|(_, output, _)| EffectId::new(output)),
                );

                self.inst(call);
            }

            Node::Theta(theta) => self.push_theta(graph, &input_values, theta),
            Node::Gamma(gamma) => self.push_gamma(graph, &input_values, gamma),
            Node::InputParam(_) | Node::OutputParam(_) | Node::Start(_) | Node::End(_) => {}
        };

        self.evaluated.insert(node.node());

        let mut outputs: Vec<_> = node
            .all_output_ports()
            .into_iter()
            .flat_map(|output| graph.get_outputs(output))
            .map(|(node, ..)| node.node())
            .collect();
        outputs.sort_unstable();
        self.evaluation_stack.extend(outputs);
    }

    pub(crate) fn push_theta(
        &mut self,
        graph: &Rvsdg,
        input_values: &BTreeMap<InputPort, Value>,
        theta: &graph::Theta,
    ) {
        let mut builder = Self {
            values: BTreeMap::new(),
            instructions: Vec::new(),
            evaluated: BTreeSet::new(),
            evaluation_stack: Vec::new(),
            top_level: false,
            inline_constants: self.inline_constants,
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
            body.push(Assign::input(input_id, value, Variance::Invariant).into());

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

            body.push(Assign::input(input_id, value, Variance::Variant { feedback_from }).into());

            if value.is_missing() {
                tracing::warn!(
                    "missing variant input value for theta body input {:?}: {:?}",
                    input,
                    param,
                );
            }

            inputs.insert(input_id, value).debug_unwrap_none();
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
            body.push(Instruction::Assign(Assign::output(output_id, value)));

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

    fn push_gamma(
        &mut self,
        graph: &Rvsdg,
        input_values: &BTreeMap<InputPort, Value>,
        gamma: &graph::Gamma,
    ) {
        // Get the gamma's conditional
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

        let (mut true_builder, mut false_builder) = (
            gamma_branch_builder(self.inline_constants),
            gamma_branch_builder(self.inline_constants),
        );
        let (mut true_block, mut false_block) = (Block::new(), Block::new());

        // Translate all of the input parameters for both branches
        for (&input, &[true_param, false_param]) in gamma.inputs().iter().zip(gamma.input_params())
        {
            gamma_input_param(
                &mut true_block,
                &mut true_builder,
                gamma.true_branch(),
                input_values,
                input,
                true_param,
                "true",
            );

            gamma_input_param(
                &mut false_block,
                &mut false_builder,
                gamma.false_branch(),
                input_values,
                input,
                false_param,
                "false",
            );
        }

        // Translate both branches into their respective builders
        true_builder.translate_into(&mut true_block, gamma.true_branch());
        false_builder.translate_into(&mut false_block, gamma.false_branch());

        // Translate the outputs for both branches
        let (mut true_outputs, mut false_outputs) = (BTreeMap::new(), BTreeMap::new());
        for (&output, &[true_param, false_param]) in
            gamma.outputs().iter().zip(gamma.output_params())
        {
            self.gamma_output_param(
                gamma.true_branch(),
                &true_builder,
                &mut true_block,
                &mut true_outputs,
                output,
                true_param,
                "true",
            );

            self.gamma_output_param(
                gamma.false_branch(),
                &false_builder,
                &mut false_block,
                &mut false_outputs,
                output,
                false_param,
                "false",
            );
        }

        // Add the output values to our current ones
        self.values.extend(true_builder.values);
        self.values.extend(false_builder.values);

        // Get the previous effect
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
            true_block.into_inner(),
            true_outputs,
            false_block.into_inner(),
            false_outputs,
            EffectId::new(gamma.effect_out()),
            prev_effect,
        ));
    }

    /// Fetches and builds an output parameter from a gamma branch
    #[allow(clippy::too_many_arguments)]
    fn gamma_output_param(
        &mut self,
        graph: &Rvsdg,
        builder: &IrBuilder,
        block: &mut Block,
        outputs: &mut BTreeMap<VarId, Value>,
        output: OutputPort,
        param: NodeId,
        branch: &str,
    ) {
        let output_param = graph.to_node::<OutputParam>(param);
        let param_source = graph.input_source(output_param.input());

        let value = builder
            .values
            .get(&param_source)
            .cloned()
            .unwrap_or(Value::Missing);

        if value.is_missing() {
            tracing::warn!(
                "missing output value for gamma {} branch {:?}: {:?}",
                branch,
                output,
                output_param,
            );
        }

        let output_id = VarId::new(output);
        block.push(Instruction::Assign(Assign::output(output_id, value)));

        // FIXME: This causes values to be overwritten since there's two branches
        self.values.insert(output, output_id.into()); // .debug_unwrap_none();
        outputs.insert(output_id, value).debug_unwrap_none();
    }
}

/// Creates an [`IrBuilder`] for gamma branches
fn gamma_branch_builder(inline_constants: bool) -> IrBuilder {
    IrBuilder {
        values: BTreeMap::new(),
        instructions: Vec::new(),
        evaluated: BTreeSet::new(),
        evaluation_stack: Vec::new(),
        top_level: false,
        inline_constants,
    }
}

/// Builds an input parameter for a gamma branch
fn gamma_input_param(
    block: &mut Block,
    builder: &mut IrBuilder,
    graph: &Rvsdg,
    input_values: &BTreeMap<InputPort, Value>,
    input: InputPort,
    param: NodeId,
    branch: &str,
) {
    let input_param = graph.to_node::<InputParam>(param);
    let input_id = VarId::new(input_param.output());

    let value = input_values.get(&input).cloned().unwrap_or(Value::Missing);
    block.push(Assign::input(input_id, value, Variance::None).into());

    if value.is_missing() {
        tracing::warn!(
            "missing input value for gamma {} branch input {:?}: {:?}",
            branch,
            input,
            input_param,
        );
    }

    builder
        .values
        .insert(input_param.output(), input_id.into())
        .debug_unwrap_none();
}
