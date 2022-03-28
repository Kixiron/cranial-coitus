use crate::{
    graph::{Eq, Gamma, InputPort, Neq, Node, NodeExt, Rvsdg, Sub, Theta},
    passes::{utils::Changes, Pass},
    utils::HashMap,
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
}

impl Pass for Equality {
    fn pass_name(&self) -> &str {
        "equality"
    }

    fn did_change(&self) -> bool {
        self.changes.has_changed()
    }

    fn reset(&mut self) {
        self.changes.reset();
    }

    fn report(&self) -> HashMap<&'static str, usize> {
        self.changes.as_map()
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
        if self.input_is_zero(graph, eq.rhs()) {
            if let Some(&sub) = graph.cast_input_source::<Sub>(eq.lhs()) {
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
        if self.input_is_zero(graph, neq.rhs()) {
            if let Some(&sub) = graph.cast_input_source::<Sub>(neq.lhs()) {
                let (lhs_src, rhs_src) =
                    (graph.input_source(sub.lhs()), graph.input_source(sub.rhs()));

                let new_neq = graph.neq(lhs_src, rhs_src);
                graph.rewire_dependents(neq.value(), new_neq.value());

                self.changes.inc::<"neq-rewrites">();
            }
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
