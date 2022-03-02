use crate::{
    graph::{Bool, Gamma, InputParam, Int, Load, NodeExt, OutputPort, Rvsdg, Theta},
    ir::Const,
    passes::Pass,
    utils::{AssertNone, HashMap},
    values::Ptr,
};

/// Deduplicates constants within the graph, reusing them as much as possible
pub struct ExprDedup {
    // TODO: Use ConstantStore
    constants: HashMap<OutputPort, Const>,
    // union_find: UnionFind,
    changed: bool,
    deduplicated_loads: usize,
}

impl ExprDedup {
    pub fn new() -> Self {
        Self {
            constants: HashMap::with_hasher(Default::default()),
            // union_find: UnionFind::new(),
            changed: false,
            deduplicated_loads: 0,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    // TODO: Finish union find deduplication
    //  fn add(&mut self, node: Node) -> Unioned {
    //      todo!()
    //  }
    //
    //  fn lookup(&mut self, node: &mut Node) -> Option<Unioned> {
    //      self.changed |= node.update_inputs(|input, _kind| {
    //
    //      });
    //  }
}

// TODO: Use a union-find to deduplicate all expressions
// TODO: Deduplicate invariant loop inputs
impl Pass for ExprDedup {
    fn pass_name(&self) -> &str {
        "expression-deduplication"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.constants.clear();
        self.changed = false;
    }

    fn report(&self) -> HashMap<&'static str, usize> {
        map! {
            "deduplicated loads" => self.deduplicated_loads,
        }
    }

    fn visit_bool(&mut self, graph: &mut Rvsdg, bool: Bool, value: bool) {
        if let Some((&const_id, _)) = self
            .constants
            .iter()
            .find(|&(_, known)| known.as_bool().map_or(false, |known| known == value))
        {
            let existing_const = graph.get_node(graph.port_parent(const_id));
            let (const_id, const_value) = existing_const.as_bool().map_or_else(
                || {
                    // Input params can also produce constant values
                    let param = existing_const.to_input_param();
                    (param.node(), param.output())
                },
                |(bool, _)| (bool.node(), bool.value()),
            );

            tracing::debug!(
                "deduplicated bool {:?}, {:?} already exists",
                bool.node(),
                const_id,
            );

            graph.rewire_dependents(bool.value(), const_value);
            graph.remove_outputs(bool.node());
            self.constants.remove(&bool.value());

            self.changed();
        } else {
            self.constants
                .insert(bool.value(), value.into())
                .debug_unwrap_none();
        }
    }

    fn visit_load(&mut self, graph: &mut Rvsdg, load: Load) {
        let _: Option<()> = try {
            let next_load = graph.get_output(load.output_effect())?.0.as_load()?;

            let current_ptr = graph.input_source(load.ptr());
            let next_ptr = graph.input_source(next_load.ptr());

            // If we see a situation like
            // ```
            // x := load ptr
            // y := load ptr
            // ```
            // we can remove the second load and replace it with the first
            if current_ptr == next_ptr {
                tracing::trace!(
                    current_load = ?load,
                    next_load = ?next_load,
                    ptr = ?current_ptr,
                    "deduplicated two loads of the same address",
                );

                graph.rewire_dependents(next_load.output_value(), load.output_value());
                graph.rewire_dependents(next_load.output_effect(), load.output_effect());
                graph.remove_node(next_load.node());

                self.deduplicated_loads += 1;
                self.changed();
            }
        };
    }

    fn visit_int(&mut self, graph: &mut Rvsdg, int: Int, value: Ptr) {
        if let Some((&const_id, _)) = self
            .constants
            .iter()
            .find(|&(_, known)| known.as_ptr().map_or(false, |known| known == value))
        {
            let existing_const = graph.get_node(graph.port_parent(const_id));
            let (const_id, const_value) = existing_const.as_int().map_or_else(
                || {
                    // Input params can also produce constant values
                    let param = existing_const.to_input_param();
                    (param.node(), param.output())
                },
                |(int, _)| (int.node(), int.value()),
            );

            tracing::debug!(
                "deduplicated int {:?}, {:?} already exists",
                int.node(),
                const_id,
            );

            graph.rewire_dependents(int.value(), const_value);
            graph.remove_outputs(int.node());
            self.constants.remove(&int.value());

            self.changed();
        } else {
            self.constants
                .insert(int.value(), value.into())
                .debug_unwrap_none();
        }
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;
        let (mut truthy_visitor, mut falsy_visitor) = (Self::new(), Self::new());

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[true_param, false_param]) in gamma.inputs().iter().zip(gamma.input_params())
        {
            if let Some(constant) = self.constants.get(&graph.input_source(input)).copied() {
                let param = gamma.true_branch().to_node::<InputParam>(true_param);
                truthy_visitor
                    .constants
                    .insert(param.output(), constant)
                    .debug_unwrap_none();

                let param = gamma.false_branch().to_node::<InputParam>(false_param);
                falsy_visitor
                    .constants
                    .insert(param.output(), constant)
                    .debug_unwrap_none();
            }
        }

        changed |= truthy_visitor.visit_graph(gamma.true_mut());
        changed |= falsy_visitor.visit_graph(gamma.false_mut());

        if changed {
            graph.replace_node(gamma.node(), gamma);
            self.changed();
        }
    }

    // TODO: There's some push/pull-based things we should do for routing constant values
    //       into regions so that we could avoid duplicating constant values within
    //       regions. However, this could have the downside of requiring more/better
    //       constant propagation as the values of constants wouldn't be immediately
    //       available. Everything's a tradeoff, the work involved with this one combined
    //       with its potential failure make it a low priority
    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut changed = false;
        let mut visitor = Self::new();

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (input, param) in theta.invariant_input_pairs() {
            if let Some(constant) = self.constants.get(&graph.input_source(input)).cloned() {
                visitor
                    .constants
                    .insert(param.output(), constant)
                    .debug_unwrap_none();
            }
        }

        changed |= visitor.visit_graph(theta.body_mut());

        // Deduplicate invariant parameters
        // TODO: Buffers
        let sources: Vec<_> = theta
            .invariant_input_pairs()
            .map(|(port, param)| (port, graph.input_source(port), param))
            .collect();
        let inputs: Vec<_> = theta.invariant_input_pairs().collect();

        for (port, param) in inputs {
            let source = graph.input_source(port);

            if theta.has_invariant_input(port) {
                if let Some((new_port, new_source, new_param)) = sources
                    .iter()
                    .find(|&&(input, parm_source, _)| {
                        theta.has_invariant_input(input) && input != port && source == parm_source
                    })
                    .copied()
                {
                    tracing::trace!(
                        old_port = ?port,
                        old_source = ?source,
                        old_param = ?param,
                        ?new_port,
                        ?new_source,
                        ?new_param,
                        ?theta,
                        "deduplicated invariant theta input {:?} with {:?}",
                        port,
                        new_port,
                    );

                    theta
                        .body_mut()
                        .rewire_dependents(param.output(), new_param.output());
                    theta.remove_invariant_input(port);
                    theta.body_mut().remove_node(param.node());

                    changed = true;
                }
            }
        }

        if changed {
            graph.replace_node(theta.node(), theta);
            self.changed();
        }
    }
}

impl Default for ExprDedup {
    fn default() -> Self {
        Self::new()
    }
}
