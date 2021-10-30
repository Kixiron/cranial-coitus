use crate::{
    graph::{Bool, Gamma, InputParam, Int, NodeExt, OutputPort, Rvsdg, Theta},
    ir::Const,
    passes::Pass,
};
use std::collections::BTreeMap;

/// Deduplicates constants within the graph, reusing them as much as possible
pub struct ConstDedup {
    constants: BTreeMap<OutputPort, Const>,
    changed: bool,
}

impl ConstDedup {
    pub fn new() -> Self {
        Self {
            constants: BTreeMap::new(),
            changed: false,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }
}

// TODO: Propagate constants through regions
impl Pass for ConstDedup {
    fn pass_name(&self) -> &str {
        "constant-deduplication"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.constants.clear();
        self.changed = false;
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
            let replaced = self.constants.insert(bool.value(), value.into());
            debug_assert!(replaced.is_none());
        }
    }

    fn visit_int(&mut self, graph: &mut Rvsdg, int: Int, value: i32) {
        if let Some((&const_id, _)) = self
            .constants
            .iter()
            .find(|&(_, known)| known.as_int().map_or(false, |known| known == value))
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
            let replaced = self.constants.insert(int.value(), value.into());
            debug_assert!(replaced.is_none());
        }
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let (mut truthy_visitor, mut falsy_visitor) = (Self::new(), Self::new());

        // For each input into the gamma region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (&input, &[true_param, false_param]) in gamma.inputs().iter().zip(gamma.input_params())
        {
            if let Some(constant) = self.constants.get(&graph.input_source(input)).cloned() {
                let param = gamma.true_branch().to_node::<InputParam>(true_param);
                let replaced = truthy_visitor
                    .constants
                    .insert(param.output(), constant.clone());
                debug_assert!(replaced.is_none());

                let param = gamma.false_branch().to_node::<InputParam>(false_param);
                let replaced = falsy_visitor.constants.insert(param.output(), constant);
                debug_assert!(replaced.is_none());
            }
        }

        truthy_visitor.visit_graph(gamma.true_mut());
        falsy_visitor.visit_graph(gamma.false_mut());
        self.changed |= truthy_visitor.did_change();
        self.changed |= falsy_visitor.did_change();

        graph.replace_node(gamma.node(), gamma);
    }

    // TODO: There's some push/pull-based things we should do for routing constant values
    //       into regions so that we could avoid duplicating constant values within
    //       regions. However, this could have the downside of requiring more/better
    //       constant propagation as the values of constants wouldn't be immediately
    //       available. Everything's a tradeoff, the work involved with this one combined
    //       with its potential failure make it a low priority
    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut visitor = Self::new();

        // For each input into the theta region, if the input value is a known constant
        // then we should associate the input value with said constant
        for (input, param) in theta.input_pairs() {
            if let Some(constant) = self.constants.get(&graph.input_source(input)).cloned() {
                let replaced = visitor.constants.insert(param.output(), constant);
                debug_assert!(replaced.is_none());
            }
        }

        visitor.visit_graph(theta.body_mut());
        self.changed |= visitor.did_change();

        graph.replace_node(theta.node(), theta);
    }
}

impl Default for ConstDedup {
    fn default() -> Self {
        Self::new()
    }
}
