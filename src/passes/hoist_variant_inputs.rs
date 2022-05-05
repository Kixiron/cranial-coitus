use crate::{
    graph::{Gamma, NodeExt, Rvsdg, Theta},
    passes::{
        utils::{ChangeReport, Changes},
        Pass,
    },
};

/// Fuses chained additions based on the law of associative addition
// TODO: Equality is also associative but it's unclear whether or not
//       that situation can actually arise within brainfuck programs
pub struct HoistVariantInputs {
    changes: Changes<1>,
}

impl HoistVariantInputs {
    pub fn new() -> Self {
        Self {
            changes: Changes::new(["variant-inputs"]),
        }
    }

    fn hoist_theta_inputs(&self, theta: &mut Theta) -> bool {
        for input in theta.input_params() {}

        todo!()
    }
}

impl Pass for HoistVariantInputs {
    fn pass_name(&self) -> &str {
        "hoist-variant-inputs"
    }

    fn did_change(&self) -> bool {
        self.changes.did_change()
    }

    fn reset(&mut self) {
        self.changes.reset();
    }

    fn report(&self) -> ChangeReport {
        self.changes.as_report()
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut changed = self.visit_graph(theta.body_mut());
        changed |= self.hoist_theta_inputs(&mut theta);

        if changed {
            graph.replace_node(theta.node(), theta);
        }
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        if self.visit_graph(gamma.true_mut()) | self.visit_graph(gamma.false_mut()) {
            graph.replace_node(gamma.node(), gamma);
        }
    }
}
