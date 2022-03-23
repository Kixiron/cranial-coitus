use crate::{
    graph::{EdgeKind, Gamma, NodeExt, Output, Rvsdg, Theta},
    passes::{utils::Changes, Pass},
    utils::HashMap,
};

pub struct FuseOutputs {
    changes: Changes<1>,
}

impl FuseOutputs {
    pub fn new() -> Self {
        Self {
            changes: Changes::new(["outputs-fused"]),
        }
    }
}

impl Pass for FuseOutputs {
    fn pass_name(&self) -> &str {
        "fuse-outputs"
    }

    fn did_change(&self) -> bool {
        self.changes.has_changed()
    }

    fn reset(&mut self) {
        self.changes.set_has_changed(false);
    }

    fn report(&self) -> HashMap<&'static str, usize> {
        self.changes.as_map()
    }

    fn visit_output(&mut self, graph: &mut Rvsdg, mut output: Output) {
        let mut changed = false;
        while let Some(consumer) = graph.cast_target::<Output>(output.output_effect()).cloned() {
            // Add each value from the fused output to the current one
            output.values_mut().reserve(consumer.values().len());
            for &value in consumer.values() {
                let source = graph.input_source(value);
                let port = graph.input_port(output.node(), EdgeKind::Value);
                graph.add_value_edge(source, port);
                output.values_mut().push(port);
            }

            // Rewire the effects from the consumer to the current output
            graph.rewire_dependents(consumer.output_effect(), output.output_effect());
            graph.remove_node(consumer.node());

            changed = true;
            self.changes.inc::<"outputs-fused">();
        }

        // If we fused any output calls together, replace the node
        if changed {
            graph.replace_node(output.node(), output);
        }
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut changed = false;
        changed |= self.visit_graph(gamma.true_mut());
        changed |= self.visit_graph(gamma.false_mut());

        if changed {
            graph.replace_node(gamma.node(), gamma);
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        if self.visit_graph(theta.body_mut()) {
            graph.replace_node(theta.node(), theta);
        }
    }
}
