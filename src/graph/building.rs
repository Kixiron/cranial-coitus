use crate::{
    graph::{
        nodes::{GammaStub, ThetaEffects, ThetaStub},
        Add, Bool, EdgeKind, End, Eq, Gamma, GammaData, Input, InputParam, Int, Load, Mul, Neg,
        Node, Not, Output, OutputParam, OutputPort, Rvsdg, Start, Store, Subgraph, Theta,
        ThetaData,
    },
    utils::AssertNone,
};
use std::collections::BTreeMap;
use tinyvec::TinyVec;

impl Rvsdg {
    pub fn start(&mut self) -> Start {
        let start_id = self.next_node();
        self.start_nodes.push(start_id);

        let effect = self.output_port(start_id, EdgeKind::Effect);

        let start = Start::new(start_id, effect);
        self.nodes
            .insert(start_id, Node::Start(start))
            .debug_unwrap_none();

        start
    }

    #[track_caller]
    pub fn end(&mut self, effect: OutputPort) -> End {
        self.assert_effect_port(effect);

        let end_id = self.next_node();
        self.end_nodes.push(end_id);

        let effect_port = self.input_port(end_id, EdgeKind::Effect);
        self.add_effect_edge(effect, effect_port);

        let end = End::new(end_id, effect_port);
        self.nodes
            .insert(end_id, Node::End(end))
            .debug_unwrap_none();

        end
    }

    pub fn int(&mut self, value: i32) -> Int {
        let int_id = self.next_node();

        let output = self.output_port(int_id, EdgeKind::Value);

        let int = Int::new(int_id, output);
        self.nodes
            .insert(int_id, Node::Int(int, value))
            .debug_unwrap_none();

        int
    }

    pub fn bool(&mut self, value: bool) -> Bool {
        let bool_id = self.next_node();

        let output = self.output_port(bool_id, EdgeKind::Value);

        let bool = Bool::new(bool_id, output);
        self.nodes
            .insert(bool_id, Node::Bool(bool, value))
            .debug_unwrap_none();

        bool
    }

    #[track_caller]
    pub fn add(&mut self, lhs: OutputPort, rhs: OutputPort) -> Add {
        self.assert_value_port(lhs);
        self.assert_value_port(rhs);

        let add_id = self.next_node();

        let lhs_port = self.input_port(add_id, EdgeKind::Value);
        self.add_value_edge(lhs, lhs_port);

        let rhs_port = self.input_port(add_id, EdgeKind::Value);
        self.add_value_edge(rhs, rhs_port);

        let output = self.output_port(add_id, EdgeKind::Value);

        let add = Add::new(add_id, lhs_port, rhs_port, output);
        self.nodes
            .insert(add_id, Node::Add(add))
            .debug_unwrap_none();

        add
    }

    #[track_caller]
    pub fn mul(&mut self, lhs: OutputPort, rhs: OutputPort) -> Mul {
        self.assert_value_port(lhs);
        self.assert_value_port(rhs);

        let mul_id = self.next_node();

        let lhs_port = self.input_port(mul_id, EdgeKind::Value);
        self.add_value_edge(lhs, lhs_port);

        let rhs_port = self.input_port(mul_id, EdgeKind::Value);
        self.add_value_edge(rhs, rhs_port);

        let output = self.output_port(mul_id, EdgeKind::Value);

        let mul = Mul::new(mul_id, lhs_port, rhs_port, output);
        self.nodes
            .insert(mul_id, Node::Mul(mul))
            .debug_unwrap_none();

        mul
    }

    #[track_caller]
    pub fn load(&mut self, ptr: OutputPort, effect: OutputPort) -> Load {
        self.assert_value_port(ptr);
        self.assert_effect_port(effect);

        let load_id = self.next_node();

        let ptr_port = self.input_port(load_id, EdgeKind::Value);
        self.add_value_edge(ptr, ptr_port);

        let effect_port = self.input_port(load_id, EdgeKind::Effect);
        self.add_effect_edge(effect, effect_port);

        let loaded = self.output_port(load_id, EdgeKind::Value);
        let effect_out = self.output_port(load_id, EdgeKind::Effect);

        let load = Load::new(load_id, ptr_port, effect_port, loaded, effect_out);
        self.nodes
            .insert(load_id, Node::Load(load))
            .debug_unwrap_none();

        load
    }

    #[track_caller]
    pub fn store(&mut self, ptr: OutputPort, value: OutputPort, effect: OutputPort) -> Store {
        self.assert_value_port(ptr);
        self.assert_value_port(value);
        self.assert_effect_port(effect);

        let store_id = self.next_node();

        let ptr_port = self.input_port(store_id, EdgeKind::Value);
        self.add_value_edge(ptr, ptr_port);

        let value_port = self.input_port(store_id, EdgeKind::Value);
        self.add_value_edge(value, value_port);

        let effect_port = self.input_port(store_id, EdgeKind::Effect);
        self.add_effect_edge(effect, effect_port);

        let effect_out = self.output_port(store_id, EdgeKind::Effect);

        let store = Store::new(store_id, ptr_port, value_port, effect_port, effect_out);
        self.nodes
            .insert(store_id, Node::Store(store))
            .debug_unwrap_none();

        store
    }

    #[track_caller]
    pub fn input(&mut self, effect: OutputPort) -> Input {
        self.assert_effect_port(effect);

        let input_id = self.next_node();

        let effect_port = self.input_port(input_id, EdgeKind::Effect);
        self.add_effect_edge(effect, effect_port);

        let value = self.output_port(input_id, EdgeKind::Value);
        let effect_out = self.output_port(input_id, EdgeKind::Effect);

        let input = Input::new(input_id, effect_port, value, effect_out);
        self.nodes
            .insert(input_id, Node::Input(input))
            .debug_unwrap_none();

        input
    }

    #[track_caller]
    pub fn output(&mut self, value: OutputPort, effect: OutputPort) -> Output {
        self.assert_value_port(value);
        self.assert_effect_port(effect);

        let output_id = self.next_node();

        let value_port = self.input_port(output_id, EdgeKind::Value);
        self.add_value_edge(value, value_port);

        let effect_port = self.input_port(output_id, EdgeKind::Effect);
        self.add_effect_edge(effect, effect_port);

        let effect_out = self.output_port(output_id, EdgeKind::Effect);

        let output = Output::new(output_id, value_port, effect_port, effect_out);
        self.nodes
            .insert(output_id, Node::Output(output))
            .debug_unwrap_none();

        output
    }

    pub fn input_param(&mut self, kind: EdgeKind) -> InputParam {
        let input_id = self.next_node();

        let port = self.output_port(input_id, EdgeKind::Value);
        let param = InputParam::new(input_id, port, kind);
        self.nodes
            .insert(input_id, Node::InputParam(param))
            .debug_unwrap_none();

        param
    }

    pub fn output_param(&mut self, input: OutputPort, kind: EdgeKind) -> OutputParam {
        let output_id = self.next_node();

        let port = self.input_port(output_id, EdgeKind::Value);
        self.add_edge(input, port, kind);

        let param = OutputParam::new(output_id, port, kind);
        self.nodes
            .insert(output_id, Node::OutputParam(param))
            .debug_unwrap_none();

        param
    }

    /// Builds a theta node
    ///
    /// `invariant_inputs` is a list of values to be given to the theta's body that
    /// *never change upon iteration*. These values will always stay the same, no matter
    /// how many times the loop iterates.
    ///
    /// `variant_inputs` is a list of values that can change upon iteration. These values
    /// are allowed to evolve as the loop iterates and are fed back into by the `outputs`
    /// field of the [`ThetaData`] constructed in the `build_theta` function.
    ///
    /// `effect` is an optional effect edge to be fed into the theta node. Thetas don't
    /// have to have an input effect, so this is optional.
    ///
    /// The `build_theta` function receives a mutable reference to the theta's body,
    /// the [`OutputPort`] from the body's start node, the [`OutputPort`]s
    /// of all invariant inputs and the [`OutputPort`]s of all variant inputs.
    /// The user doesn't need to create [`Start`] or [`End`] nodes for the theta's
    /// body, these are handled automatically. While it may seem odd that [`ThetaData`]
    /// always requires an [`OutputPort`] for the outgoing effect, this is because
    /// the theta's body still has effect edge flow regardless of whether or not the
    /// outer theta has a incoming/outgoing effects. However, if the outer theta has
    /// no incoming or outgoing effect edges the theta's body effects should be a
    /// direct connection between the body's [`Start`] and [`End`] nodes. That is,
    /// the [`OutputPort`] passed out of the `build_theta` function should be the same
    /// one that was passed to the `build_theta` function as the effect parameter.
    /// Finally, the `condition` of the [`ThetaData`] should be an expression that
    /// evaluates to a boolean value, this is the exit condition of the theta node.
    ///
    /// **The ordering of `invariant_inputs`, `variant_inputs` and `outputs` on the produced
    /// `ThetaData` are all important!!!**
    /// The order of these collections are used to associate things together, the nth element
    /// of the `variant_inputs` parameter will be associated with nth elements of both the
    /// `variant_inputs` slice given to the `build_theta` function and the `outputs` field
    /// of the produced `ThetaData`!
    ///
    // TODO: Refactor this
    pub fn theta<I1, I2, E, F>(
        &mut self,
        invariant_inputs: I1,
        variant_inputs: I2,
        effect: E,
        build_theta: F,
    ) -> ThetaStub
    where
        I1: IntoIterator<Item = OutputPort>,
        I2: IntoIterator<Item = OutputPort>,
        E: Into<Option<OutputPort>>,
        F: FnOnce(&mut Rvsdg, OutputPort, &[OutputPort], &[OutputPort]) -> ThetaData,
    {
        // Create the theta's node id
        let theta_id = self.next_node();

        // If an input effect was given, create a port for it
        let effect_source = effect.into();
        let effect_input = effect_source.map(|effect_source| {
            self.assert_effect_port(effect_source);

            let effect_input = self.input_port(theta_id, EdgeKind::Effect);
            self.add_effect_edge(effect_source, effect_input);

            effect_input
        });

        // Create input ports for the given invariant inputs
        let invariant_input_ports: TinyVec<[_; 5]> = invariant_inputs
            .into_iter()
            .map(|input| {
                self.assert_value_port(input);

                let port = self.input_port(theta_id, EdgeKind::Value);
                self.add_value_edge(input, port);

                port
            })
            .collect();

        // Create input ports for the given variant inputs
        let variant_input_ports: TinyVec<[_; 5]> = variant_inputs
            .into_iter()
            .map(|input| {
                self.assert_value_port(input);

                let port = self.input_port(theta_id, EdgeKind::Value);
                self.add_value_edge(input, port);

                port
            })
            .collect();

        // Create the theta's subgraph
        let mut subgraph =
            Rvsdg::from_counters(self.node_counter.clone(), self.port_counter.clone());

        // Create the theta start node
        let start = subgraph.start();

        // Create the input params for the invariant inputs
        let (invariant_inputs, invariant_param_outputs): (BTreeMap<_, _>, TinyVec<[_; 5]>) =
            invariant_input_ports
                .iter()
                .map(|&input| {
                    let param = subgraph.input_param(EdgeKind::Value);
                    ((input, param.node()), param.output())
                })
                .unzip();

        // Create the input params for the variant inputs
        let (variant_inputs, variant_param_outputs): (BTreeMap<_, _>, TinyVec<[_; 5]>) =
            variant_input_ports
                .iter()
                .map(|&input| {
                    let param = subgraph.input_param(EdgeKind::Value);
                    ((input, param.node()), param.output())
                })
                .unzip();

        // Build the theta node's body
        let ThetaData {
            outputs,
            condition,
            effect: body_effect_output,
        } = build_theta(
            &mut subgraph,
            start.effect(),
            &invariant_param_outputs,
            &variant_param_outputs,
        );

        // Create the subgraph condition's output param
        subgraph.assert_value_port(condition);
        let condition_param = subgraph.output_param(condition, EdgeKind::Value);

        // Create the subgraph's end node
        subgraph.assert_effect_port(body_effect_output);
        let end = subgraph.end(body_effect_output);

        // If there's no input effect then the body can't contain effectful operations
        if effect_input.is_none() {
            assert_eq!(
                body_effect_output,
                start.effect(),
                "if the theta node isn't connected to effect flow, \
                the body cannot have effectful operations",
            );
        }

        // Make sure every variant input has a paired output
        assert_eq!(
            variant_inputs.len(),
            outputs.len(),
            "theta nodes must have the same number of outputs as there are variant inputs",
        );

        // Create the output params for all outputs from the body
        let output_params =
            outputs
                .iter()
                .zip(variant_inputs.keys())
                .map(|(&output, &variant_input)| {
                    subgraph.assert_value_port(output);

                    // Create the output param within the subgraph
                    let output_param = subgraph.output_param(output, EdgeKind::Value);

                    (variant_input, output_param)
                });

        // Create the map of theta output ports to subgraph input params and
        // the map of back edges between output ports and variant inputs
        let (outputs, output_back_edges): (BTreeMap<_, _>, BTreeMap<_, _>) = output_params
            .map(|(variant_input, output_param)| {
                // Create the output port on the theta node
                let output_port = self.output_port(theta_id, EdgeKind::Value);

                (
                    (output_port, output_param.node()),
                    (output_port, variant_input),
                )
            })
            .unzip();
        let output_ports: TinyVec<[OutputPort; 5]> = outputs.keys().copied().collect();

        // If we were given an input effect then we need to make an output effect as well
        let effects = effect_input.map(|effect_input| {
            let effect_output = self.output_port(theta_id, EdgeKind::Effect);

            ThetaEffects::new(effect_input, effect_output)
        });

        let theta = Theta::new(
            theta_id,
            effects,
            invariant_inputs,
            variant_inputs,
            outputs,
            output_back_edges,
            condition_param.node(),
            Box::new(Subgraph::new(subgraph, start.node(), end.node())),
        );

        let stub = ThetaStub::new(effects.map(|effect| effect.output()), output_ports);
        self.nodes
            .insert(theta_id, Node::Theta(Box::new(theta)))
            .debug_unwrap_none();

        stub
    }

    // TODO: Refactor this
    #[track_caller]
    pub fn gamma<I, T, F>(
        &mut self,
        inputs: I,
        effect: OutputPort,
        condition: OutputPort,
        truthy: T,
        falsy: F,
    ) -> GammaStub
    where
        I: IntoIterator<Item = OutputPort>,
        T: FnOnce(&mut Rvsdg, OutputPort, &[OutputPort]) -> GammaData,
        F: FnOnce(&mut Rvsdg, OutputPort, &[OutputPort]) -> GammaData,
    {
        self.assert_effect_port(effect);
        self.assert_value_port(condition);

        let gamma_id = self.next_node();

        let effect_in = self.input_port(gamma_id, EdgeKind::Effect);
        self.add_effect_edge(effect, effect_in);

        let cond_port = self.input_port(gamma_id, EdgeKind::Value);
        self.add_value_edge(condition, cond_port);

        // Wire up the external inputs to the gamma node
        let outer_inputs: TinyVec<[_; 4]> = inputs
            .into_iter()
            .map(|input| {
                self.assert_value_port(input);

                let port = self.input_port(gamma_id, EdgeKind::Value);
                self.add_value_edge(input, port);
                port
            })
            .collect();

        // Create the gamma's true branch
        let mut truthy_subgraph =
            Rvsdg::from_counters(self.node_counter.clone(), self.port_counter.clone());

        // Create the input ports within the subgraph
        let (truthy_input_params, truthy_inner_input_ports): (TinyVec<[_; 4]>, TinyVec<[_; 4]>) =
            (0..outer_inputs.len())
                .map(|_| {
                    let param = truthy_subgraph.input_param(EdgeKind::Value);
                    (param.node(), param.output())
                })
                .unzip();

        // Create the branch's start node
        let truthy_start = truthy_subgraph.start();
        let GammaData {
            outputs: truthy_outputs,
            effect: truthy_output_effect,
        } = truthy(
            &mut truthy_subgraph,
            truthy_start.effect(),
            &truthy_inner_input_ports,
        );

        truthy_subgraph.assert_effect_port(truthy_output_effect);
        let truthy_end = truthy_subgraph.end(truthy_output_effect);

        let truthy_output_params: TinyVec<[_; 4]> = truthy_outputs
            .iter()
            .map(|&output| {
                truthy_subgraph.assert_value_port(output);
                truthy_subgraph.output_param(output, EdgeKind::Value).node()
            })
            .collect();

        // Create the gamma's true branch
        let mut falsy_subgraph =
            Rvsdg::from_counters(self.node_counter.clone(), self.port_counter.clone());

        // Create the input ports within the subgraph
        let (falsy_input_params, falsy_inner_input_ports): (TinyVec<[_; 4]>, TinyVec<[_; 4]>) = (0
            ..outer_inputs.len())
            .map(|_| {
                let param = falsy_subgraph.input_param(EdgeKind::Value);
                (param.node(), param.output())
            })
            .unzip();

        // Create the branch's start node
        let falsy_start = falsy_subgraph.start();
        let GammaData {
            outputs: falsy_outputs,
            effect: falsy_output_effect,
        } = falsy(
            &mut falsy_subgraph,
            falsy_start.effect(),
            &falsy_inner_input_ports,
        );

        falsy_subgraph.assert_effect_port(falsy_output_effect);
        let falsy_end = falsy_subgraph.end(falsy_output_effect);

        let falsy_output_params: TinyVec<[_; 4]> = falsy_outputs
            .iter()
            .map(|&output| {
                falsy_subgraph.assert_value_port(output);
                falsy_subgraph.output_param(output, EdgeKind::Value).node()
            })
            .collect();

        // FIXME: I'd really like to be able to support variable numbers of inputs for each branch
        //        to allow some more flexible optimizations like removing effect flow from a branch
        debug_assert_eq!(truthy_input_params.len(), falsy_input_params.len());
        // FIXME: Remove the temporary allocations
        let input_params = truthy_input_params
            .into_iter()
            .zip(falsy_input_params)
            .map(|(truthy, falsy)| [truthy, falsy])
            .collect();

        debug_assert_eq!(truthy_output_params.len(), falsy_output_params.len());
        // FIXME: Remove the temporary allocations
        let output_params: TinyVec<[_; 4]> = truthy_output_params
            .into_iter()
            .zip(falsy_output_params)
            .map(|(truthy, falsy)| [truthy, falsy])
            .collect();

        let effect_out = self.output_port(gamma_id, EdgeKind::Effect);
        let outer_outputs: TinyVec<[_; 4]> = (0..output_params.len())
            .map(|_| self.output_port(gamma_id, EdgeKind::Value))
            .collect();

        let stub = GammaStub::new(Some(effect_out), outer_outputs.iter().copied().collect());
        let gamma = Gamma::new(
            gamma_id,                                    // node
            outer_inputs,                                // inputs
            effect_in,                                   // effect_in
            input_params,                                // input_params
            falsy_start.effect(),                        // input_effect
            outer_outputs,                               // outputs
            effect_out,                                  // effect_out
            output_params,                               // output_params
            [truthy_output_effect, falsy_output_effect], // output_effect
            [truthy_start.node(), falsy_start.node()],   // start_nodes
            [truthy_end.node(), falsy_end.node()],       // end_nodes
            Box::new([truthy_subgraph, falsy_subgraph]), // body
            cond_port,                                   // condition
        );

        self.nodes
            .insert(gamma_id, Node::Gamma(Box::new(gamma)))
            .debug_unwrap_none();

        stub
    }

    #[track_caller]
    pub fn eq(&mut self, lhs: OutputPort, rhs: OutputPort) -> Eq {
        self.assert_value_port(lhs);
        self.assert_value_port(rhs);

        let eq_id = self.next_node();

        let lhs_port = self.input_port(eq_id, EdgeKind::Value);
        self.add_value_edge(lhs, lhs_port);

        let rhs_port = self.input_port(eq_id, EdgeKind::Value);
        self.add_value_edge(rhs, rhs_port);

        let output = self.output_port(eq_id, EdgeKind::Value);

        let eq = Eq::new(eq_id, lhs_port, rhs_port, output);
        self.nodes.insert(eq_id, Node::Eq(eq)).debug_unwrap_none();

        eq
    }

    #[track_caller]
    pub fn not(&mut self, input: OutputPort) -> Not {
        self.assert_value_port(input);

        let not_id = self.next_node();

        let input_port = self.input_port(not_id, EdgeKind::Value);
        self.add_value_edge(input, input_port);

        let output = self.output_port(not_id, EdgeKind::Value);

        let not = Not::new(not_id, input_port, output);
        self.nodes
            .insert(not_id, Node::Not(not))
            .debug_unwrap_none();

        not
    }

    #[track_caller]
    pub fn neg(&mut self, input: OutputPort) -> Neg {
        self.assert_value_port(input);

        let neg_id = self.next_node();

        let input_port = self.input_port(neg_id, EdgeKind::Value);
        self.add_value_edge(input, input_port);

        let output = self.output_port(neg_id, EdgeKind::Value);

        let neg = Neg::new(neg_id, input_port, output);
        self.nodes
            .insert(neg_id, Node::Neg(neg))
            .debug_unwrap_none();

        neg
    }

    #[track_caller]
    pub fn neq(&mut self, lhs: OutputPort, rhs: OutputPort) -> Not {
        self.assert_value_port(lhs);
        self.assert_value_port(rhs);

        let eq = self.eq(lhs, rhs);
        self.not(eq.value())
    }
}
