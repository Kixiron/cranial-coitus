use crate::{
    graph::{
        AddOrSub, End, Gamma, InputParam, Load, Neq, NodeExt, OutputParam, Rvsdg, ScanDirection,
        Theta,
    },
    ir::Const,
    passes::{utils::Changes, Pass},
    utils::HashMap,
};

pub struct ScanLoops {
    changes: Changes<2>,
}

impl ScanLoops {
    pub fn new() -> Self {
        Self {
            changes: Changes::new(["scanr", "scanl"]),
        }
    }

    fn try_rewrite_theta_scan(&mut self, graph: &mut Rvsdg, theta: &mut Theta) -> Option<()> {
        // There should only be a single variant feedback input/output pair and should have effects
        // Note that this allows any number of *invariant* inputs
        if theta.variant_inputs_len() != 1 || theta.outputs_len() != 1 || !theta.has_effects() {
            return None;
        }

        let body = theta.body();

        // Get the theta body's start node
        let start = theta.start_node();

        // The next (and only) effect within the loop should be the
        // load of the offset pointer
        //
        // ptr_value := load offset_ptr
        let ptr_value = body.cast_output_dest::<Load>(start.effect())?;
        // Ensure the load is the last effect in the body
        let _ = body.cast_output_dest::<End>(ptr_value.output_effect())?;

        // Get the offset pointer (either add or sub)
        //
        // offset_ptr := add ptr, step
        // offset_ptr := sub ptr, step
        let offset_ptr = AddOrSub::cast_input_source(body, ptr_value.ptr())?;
        // FIXME: There's no real constraint on the order of the operands if `offset_ptr` is an add
        let (ptr, step) = (offset_ptr.lhs(), offset_ptr.rhs());

        // FIXME: We don't have to only accept constants as the step value
        let step_node = body.input_source_node(step);
        if !step_node.is_const_number() {
            return None;
        }

        // ptr := in <input>
        let ptr_source = body.cast_input_source::<InputParam>(ptr)?;
        // Ensure the pointer is a variant input
        if !theta.contains_variant_input_param(ptr_source) {
            return None;
        }

        // value_is_needle := cmp.neq ptr_value, needle
        let value_is_needle = body.cast_output_dest::<Neq>(ptr_value.output_value())?;
        let needle = if body.input_source(value_is_needle.lhs()) == ptr_value.output_value() {
            value_is_needle.rhs()
        } else {
            value_is_needle.lhs()
        };

        // FIXME: We don't have to only accept constants as the needle value
        let needle_node = body.input_source_node(needle);
        if !needle_node.is_const_number() {
            return None;
        }

        // Ensure the neq is the condition of the theta node
        if body
            .cast_output_dest::<OutputParam>(value_is_needle.value())?
            .node()
            != theta.condition_id()
        {
            return None;
        }

        // ptr_feedback := out offset_ptr
        let ptr_feedback = body.cast_output_dest::<OutputParam>(offset_ptr.value())?;
        if !theta.output_feeds_back_to(ptr_feedback, ptr_source) {
            return None;
        }

        // The theta node matches our motif, now we can rewrite it into a scan call

        // Get the scan's direction
        let direction = if offset_ptr.is_add() {
            ScanDirection::Forward
        } else {
            ScanDirection::Backward
        };

        // Get the params for the scan call
        let ptr = graph.input_source(theta.variant_input_source(ptr_source)?);
        let step = graph
            .constant(
                step_node
                    .as_byte_value()
                    .map(Const::Cell)
                    .or_else(|| step_node.as_int_value().map(Const::Ptr))?,
            )
            .value();
        let needle = graph
            .constant(
                needle_node
                    .as_byte_value()
                    .map(Const::Cell)
                    .or_else(|| needle_node.as_int_value().map(Const::Ptr))?,
            )
            .value();
        let input_effect = graph.input_source(theta.input_effect()?);

        // Create the scan node
        let scan = graph.scan(direction, ptr, step, needle, input_effect);
        let (scan_output_effect, scan_output_ptr) = (scan.output_effect(), scan.output_ptr());

        // Rewire dependents and remove the theta we've replaced
        let (output_effect, ptr_output) = (theta.output_effect()?, theta.output_ports().next()?);
        graph.rewire_dependents(output_effect, scan_output_effect);
        graph.rewire_dependents(ptr_output, scan_output_ptr);
        graph.remove_node(theta.node());

        // Increment the counter on the change we performed
        match direction {
            ScanDirection::Forward => self.changes.inc::<"scanl">(),
            ScanDirection::Backward => self.changes.inc::<"scanr">(),
        }

        Some(())
    }
}

impl Pass for ScanLoops {
    fn pass_name(&self) -> &str {
        "scan-loops"
    }

    fn did_change(&self) -> bool {
        self.changes.has_changed()
    }

    fn reset(&mut self) {
        self.changes.reset()
    }

    fn report(&self) -> HashMap<&'static str, usize> {
        self.changes.as_map()
    }

    // ```
    // do {
    //     ptr := in <input> // feedback from `ptr_feedback`
    //
    //     // Offset the pointer
    //     offset_ptr := add ptr, step
    //     // Or sub for a reverse scan
    //     offset_ptr := sub ptr, step
    //
    //     // Load the value at the offset pointer
    //     ptr_value := load offset_ptr
    //
    //     // If the loaded value is the needle, finish the loop
    //     value_is_needle := cmp.neq ptr_value, needle
    //
    //     // Feed the offset pointer to the next loop iteration
    //     ptr_feedback := out offset_ptr
    // } while { value_is_needle }
    // ```
    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        // If the scan rewrite doesn't succeed and we then change the inner subgraph,
        // replace the theta node within the current graph
        if self.try_rewrite_theta_scan(graph, &mut theta).is_none()
            && self.visit_graph(theta.body_mut())
        {
            graph.replace_node(theta.node(), theta);
        }
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        // Visit both branches of the gamma, replacing it if anything changes
        if self.visit_graph(gamma.true_mut()) | self.visit_graph(gamma.false_mut()) {
            graph.replace_node(gamma.node(), gamma);
        }
    }
}
