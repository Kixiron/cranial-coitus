use crate::{
    graph::{EdgeKind, Gamma, Input, NodeExt, Output, Rvsdg, Theta},
    passes::{
        utils::{ChangeReport, Changes},
        Pass,
    },
};

pub struct FuseIO {
    changes: Changes<2>,
}

impl FuseIO {
    pub fn new() -> Self {
        Self {
            changes: Changes::new(["inputs-fused", "outputs-fused"]),
        }
    }
}

impl Pass for FuseIO {
    fn pass_name(&self) -> &'static str {
        "fuse-io"
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

    // FIXME: Would need to return some sort of aggregate type from the input
    // FIXME: Loads and stores interspersed between input calls stops them fusing, e.g.
    // ```
    // a := input()
    // store y, z
    // b := input()
    // ```
    // even though the actual effects are unrelated to each other, maybe this should
    // be addressed by separating the io and memory effect streams
    fn visit_input(&mut self, _graph: &mut Rvsdg, mut _input: Input) {
        // let mut changed = false;
        // // If the node is an output, fuse 'em
        // while let Some(consumer) = graph.cast_target::<Input>(input.output_effect()).cloned() {
        //     // Add each value from the fused output to the current one
        //     output.values_mut().reserve(consumer.values().len());
        //     for &value in consumer.values() {
        //         let source = graph.input_source(value);
        //         let port = graph.create_input_port(output.node(), EdgeKind::Value);
        //         graph.add_value_edge(source, port);
        //         output.values_mut().push(port);
        //     }
        //
        //     // Rewire the effects from the consumer to the current output
        //     graph.rewire_dependents(consumer.output_effect(), output.output_effect());
        //     graph.remove_node(consumer.node());
        //
        //     changed = true;
        //     self.changes.inc::<"outputs-fused">();
        // }
        //
        // // If we fused any output calls together, replace the node
        // if changed {
        //     graph.replace_node(output.node(), output);
        // }
    }

    // FIXME: Loads and stores interspersed between output calls stops them fusing, e.g.
    // ```
    // output(x)
    // store y, z
    // output(x)
    // ```
    // even though the actual effects are unrelated to each other, maybe this should
    // be addressed by separating the io and memory effect streams
    fn visit_output(&mut self, graph: &mut Rvsdg, mut output: Output) {
        let mut changed = false;
        // If the node is an output, fuse 'em
        while let Some(consumer) = graph.cast_target::<Output>(output.output_effect()).cloned() {
            // Add each value from the fused output to the current one
            output.values_mut().reserve(consumer.values().len());
            for &value in consumer.values() {
                let source = graph.input_source(value);
                let port = graph.create_input_port(output.node(), EdgeKind::Value);
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
