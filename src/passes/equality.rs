use crate::{
    graph::{Eq, Gamma, InputPort, Neq, Node, NodeExt, Rvsdg, Sub, Theta},
    passes::{
        utils::{ChangeReport, Changes},
        Pass,
    },
};

pub struct Equality {
    changes: Changes<2>,
}

impl Equality {
    pub fn new() -> Self {
        Self {
            changes: Changes::new(["eq-rewrites", "neq-rewrites"]),
        }
    }

    fn input_is_zero(&self, graph: &Rvsdg, input: InputPort) -> bool {
        match graph.input_source_node(input) {
            Node::Byte(_, byte) => byte.is_zero(),
            Node::Int(_, int) => int.is_zero(),
            _ => false,
        }
    }

    // We only want to perform these opts when there's exactly one consumer of the
    // input expression and that consumer is us, otherwise we duplicate expressions
    // that otherwise wouldn't need to be duplicated
    fn input_is_exclusive(&self, graph: &mut Rvsdg, input: InputPort) -> bool {
        graph.total_output_consumers(graph.input_source(input)) == 1
    }
}

impl Pass for Equality {
    fn pass_name(&self) -> &str {
        "equality"
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

    // Matches the motif of
    // ```
    // y_minus_z = sub y, z
    // y_eq_z = eq y_minus_z, 0
    // ```
    // and turns it into
    // ```
    // y_eq_z = eq y, z
    // ```
    //
    // ```py
    // s = z3.Solver();
    // s.add((x - y) == 0, x != y)
    // print(s.check()) # unsat
    // ```
    fn visit_eq(&mut self, graph: &mut Rvsdg, eq: Eq) {
        if self.input_is_zero(graph, eq.rhs()) && self.input_is_exclusive(graph, eq.lhs()) {
            if let Some(&sub) = graph.cast_input_source::<Sub>(eq.lhs()) {
                let (lhs_src, rhs_src) =
                    (graph.input_source(sub.lhs()), graph.input_source(sub.rhs()));

                let new_eq = graph.eq(lhs_src, rhs_src);
                graph.rewire_dependents(eq.value(), new_eq.value());

                self.changes.inc::<"eq-rewrites">();
            }
        } else if self.input_is_zero(graph, eq.lhs()) && self.input_is_exclusive(graph, eq.rhs()) {
            if let Some(&sub) = graph.cast_input_source::<Sub>(eq.rhs()) {
                let (lhs_src, rhs_src) =
                    (graph.input_source(sub.lhs()), graph.input_source(sub.rhs()));

                let new_eq = graph.eq(lhs_src, rhs_src);
                graph.rewire_dependents(eq.value(), new_eq.value());

                self.changes.inc::<"eq-rewrites">();
            }
        }
    }

    // Matches the motif of
    // ```
    // y_minus_z = sub y, z
    // y_neq_z = neq y_minus_z, 0
    // ```
    // and turns it into
    // ```
    // y_neq_z = neq y, z
    // ```
    //
    // ```py
    // s = z3.Solver();
    // s.add((x - y) != 0, x == y)
    // print(s.check()) # unsat
    // ```
    fn visit_neq(&mut self, graph: &mut Rvsdg, neq: Neq) {
        if self.input_is_zero(graph, neq.rhs()) && self.input_is_exclusive(graph, neq.lhs()) {
            if let Some(&sub) = graph.cast_input_source::<Sub>(neq.lhs()) {
                let (lhs_src, rhs_src) =
                    (graph.input_source(sub.lhs()), graph.input_source(sub.rhs()));

                let new_neq = graph.neq(lhs_src, rhs_src);
                graph.rewire_dependents(neq.value(), new_neq.value());

                self.changes.inc::<"neq-rewrites">();
            }
        } else if self.input_is_zero(graph, neq.lhs()) && self.input_is_exclusive(graph, neq.rhs())
        {
            if let Some(&sub) = graph.cast_input_source::<Sub>(neq.rhs()) {
                let (lhs_src, rhs_src) =
                    (graph.input_source(sub.lhs()), graph.input_source(sub.rhs()));

                let new_eq = graph.eq(lhs_src, rhs_src);
                graph.rewire_dependents(neq.value(), new_eq.value());

                self.changes.inc::<"neq-rewrites">();
            }
        }
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        if self.visit_graph(gamma.true_mut()) | self.visit_graph(gamma.false_mut()) {
            graph.replace_node(gamma.node(), gamma);
        }
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        if self.visit_graph(theta.body_mut()) {
            graph.replace_node(theta.node(), theta);
        }
    }
}
