use crate::{
    graph::{Bool, EdgeKind, Gamma, InputPort, Int, Node, NodeId, Rvsdg, Theta},
    ir::Const,
    passes::Pass,
};
use std::collections::BTreeMap;

pub struct ZeroLoop {
    changed: bool,
    values: BTreeMap<NodeId, Const>,
}

impl ZeroLoop {
    pub fn new() -> Self {
        Self {
            changed: false,
            values: BTreeMap::new(),
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    /// See if the theta node matches our zero loop patterns
    ///
    /// Zero via overflow:
    ///
    /// ```
    /// do {
    ///   _cell_value = load _cell_ptr
    ///   _plus_one = add _cell, int 1
    ///   store _cell_ptr, _plus_one
    ///   _is_zero = eq _plus_one, int 0
    ///   _not_zero = not _is_zero
    /// } while { _not_zero }
    /// ```
    ///
    /// Zero via subtraction:
    ///
    /// ```
    /// do {
    ///   _cell_value = load _cell_ptr
    ///   _plus_one = add _cell_value, int -1
    ///   store _cell_ptr, _plus_one
    ///   _is_zero = eq _plus_one, int 0
    ///   _not_zero = not _is_zero
    /// } while { _not_zero }
    /// ```
    ///
    /// In both of these cases we can replace the entire loop with a simple
    /// zero store
    ///
    /// ```
    /// store _cell_ptr, int 0
    /// ```
    fn is_zero_loop(
        &self,
        theta: &Theta,
        body_values: &BTreeMap<NodeId, Const>,
    ) -> Option<Result<i32, InputPort>> {
        let graph = theta.body();

        // The theta's start node
        let start = graph.get_node(theta.start()).to_start();

        // The node after the effect must be a load
        let load = graph.get_output(start.effect())?.0.as_load()?;

        let target_node = graph.get_input(load.ptr()).0;

        // We can zero out constant addresses or a dynamically generated one
        let target_ptr = if let Some(ptr) = body_values.get(&target_node.node_id()).cloned() {
            Ok(ptr.convert_to_i32()?)
        } else {
            // FIXME: We're pretty strict right now and only take inputs as "dynamic addresses",
            //        is it possible that this could be relaxed?
            let input = target_node.as_input_param()?;

            // Find the port that the input param refers to
            let input_port = theta
                .input_params()
                .iter()
                .zip(theta.inputs())
                .find_map(|(&param, &port)| (param == input.node()).then(|| port))?;

            Err(input_port)
        };

        // Get the add node
        let add = graph.get_output(load.value())?.0.as_add()?;

        let [lhs, rhs] = [graph.get_input(add.lhs()), graph.get_input(add.rhs())];

        // Make sure that one of the add's operands is the loaded cell and the other is 1 or -1
        if lhs.1 == load.value() {
            let value = body_values.get(&rhs.0.node_id())?.convert_to_i32()?;

            if !matches!(value, 1 | -1) {
                // TODO: If `value == 0` this loop is infinite
                return None;
            }
        } else if rhs.1 == load.value() {
            let value = body_values.get(&lhs.0.node_id())?.convert_to_i32()?;

            if !matches!(value, 1 | -1) {
                // TODO: If `value == 0` this loop is infinite
                return None;
            }
        } else {
            return None;
        }

        let store = graph.get_output(load.effect())?.0.as_store()?;
        if graph.get_input(store.value()).1 != add.value() {
            return None;
        }

        let eq = graph
            .outputs(add.node())
            .find_map(|(_, data)| data.and_then(|(node, _, _)| node.is_eq().then(|| node)))?
            .as_eq()?;

        let [lhs, rhs] = [graph.get_input(eq.lhs()), graph.get_input(eq.rhs())];

        // Make sure that one of the eq's operands is the added val and the other is 0
        if lhs.1 == add.value() {
            let value = body_values.get(&rhs.0.node_id())?.convert_to_i32()?;

            if value != 0 {
                return None;
            }
        } else if rhs.1 == add.value() {
            let value = body_values.get(&lhs.0.node_id())?.convert_to_i32()?;

            if value != 0 {
                return None;
            }
        } else {
            return None;
        }

        let not = graph.get_output(eq.value())?.0.as_not()?;

        // Make sure the `(value + 1) != 0` expression is the theta's condition
        if graph.get_output(not.value())?.0.node_id() != theta.condition() {
            return None;
        }

        let _end = graph.get_output(store.effect())?.0.as_end()?;

        Some(target_ptr)
    }
}

impl Pass for ZeroLoop {
    fn pass_name(&self) -> &str {
        "zero-loop"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.values.clear();
        self.changed = false;
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        let replaced = self.values.insert(bool.node(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Bool(value)));
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: i32) {
        let replaced = self.values.insert(int.node(), value.into());
        debug_assert!(replaced.is_none() || replaced == Some(Const::Int(value)));
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let (mut truthy_visitor, mut falsy_visitor) = (Self::new(), Self::new());

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[truthy_param, falsy_param]) in
            gamma.inputs().iter().zip(gamma.input_params())
        {
            let (input_node, _, _) = graph.get_input(input);
            let input_node_id = input_node.node_id();

            if let Some(constant) = self.values.get(&input_node_id).cloned() {
                let replaced = truthy_visitor.values.insert(truthy_param, constant.clone());
                debug_assert!(replaced.is_none());

                let replaced = falsy_visitor.values.insert(falsy_param, constant);
                debug_assert!(replaced.is_none());
            }
        }

        truthy_visitor.visit_graph(gamma.truthy_mut());
        falsy_visitor.visit_graph(gamma.falsy_mut());
        self.changed |= truthy_visitor.did_change();
        self.changed |= falsy_visitor.did_change();

        // TODO: If a gamma is equivalent to this
        // ```
        // _value = load _ptr
        // _eq = eq _value, int 0
        // _neq = not _eq
        // if _eq {
        //
        // } else {
        //   store _ptr, int 0
        // }
        // ```
        //
        // Transform it into an unconditional zero store
        //
        // ```
        // store _ptr, int 0
        // ```

        graph.replace_node(gamma.node(), gamma);
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut visitor = Self::new();

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &param) in theta.inputs().iter().zip(theta.input_params()) {
            let (input_node, _, _) = graph.get_input(input);
            let input_node_id = input_node.node_id();

            if let Some(constant) = self.values.get(&input_node_id).cloned() {
                let replaced = visitor.values.insert(param, constant);
                debug_assert!(replaced.is_none());
            }
        }

        visitor.visit_graph(theta.body_mut());
        self.changed |= visitor.did_change();

        if let Some(target_ptr) = self.is_zero_loop(&theta, &visitor.values) {
            tracing::debug!(
                "detected that theta {:?} is a zero loop, replacing with a store to {:?}",
                theta.node(),
                target_ptr,
            );

            // Get the theta's effect input
            let (_, effect_source, _) = graph.get_input(theta.effect_in());

            let target_cell = match target_ptr {
                Ok(constant) => graph.int(constant).value(),
                Err(input_param) => graph.get_input(input_param).1,
            };

            // Create the zero store
            let zero = graph.int(0);
            let store = graph.store(target_cell, zero.value(), effect_source);

            // Rewire the theta's ports
            graph.rewire_dependents(theta.effect_out(), store.effect());

            for (&input_port, &param) in theta.inputs().iter().zip(theta.input_params()) {
                let param = theta.body().get_node(param).to_input_param();

                if let Some((Node::OutputPort(output), _, EdgeKind::Value)) =
                    theta.body().get_output(param.value())
                {
                    let output_port = theta.outputs().iter().zip(theta.output_params()).find_map(
                        |(&output_port, &param)| (param == output.node()).then(|| output_port),
                    );

                    if let Some(output_port) = output_port {
                        tracing::debug!(
                            "splicing theta input to output passthrough {:?}->{:?}",
                            input_port,
                            output_port,
                        );

                        graph.splice_ports(input_port, output_port);
                    }
                }
            }

            graph.remove_node(theta.node());

            self.changed();
        } else {
            graph.replace_node(theta.node(), theta);
        }
    }
}

impl Default for ZeroLoop {
    fn default() -> Self {
        Self::new()
    }
}
