use crate::{
    graph::{EdgeKind, NodeExt, Output, Rvsdg},
    passes::Pass,
    utils::HashMap,
};

pub struct FuseOutputs {
    changed: bool,
    outputs_fused: usize,
}

impl FuseOutputs {
    pub fn new() -> Self {
        Self {
            changed: false,
            outputs_fused: 0,
        }
    }

    pub fn changed(&mut self) {
        self.changed = true;
    }
}

impl Pass for FuseOutputs {
    fn pass_name(&self) -> &str {
        "fuse-outputs"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.changed = false;
    }

    fn report(&self) -> HashMap<&'static str, usize> {
        map! {
            "outputs fused" => self.outputs_fused,
        }
    }

    fn visit_output(&mut self, graph: &mut Rvsdg, mut output: Output) {
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

            self.outputs_fused += 1;
            self.changed();
        }

        // If we fused any output calls together, replace the node
        if self.changed {
            graph.replace_node(output.node(), output);
        }
    }
}
