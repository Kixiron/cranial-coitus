use crate::{
    graph::{Add, Bool, Eq, Int, OutputPort, Rvsdg},
    passes::Pass,
};
use std::collections::BTreeSet;

#[derive(Debug)]
pub struct Canonicalize {
    constants: BTreeSet<OutputPort>,
    changed: bool,
    canonicalizations: usize,
}

impl Canonicalize {
    pub fn new() -> Self {
        Self {
            constants: BTreeSet::new(),
            changed: false,
            canonicalizations: 0,
        }
    }

    fn changed(&mut self) {
        self.changed = true;
    }
}

impl Pass for Canonicalize {
    fn pass_name(&self) -> &str {
        "canonicalize"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.constants.clear();
    }

    fn report(&self) -> crate::utils::HashMap<&'static str, usize> {
        map! {
            "canonicalizations" => self.canonicalizations,
        }
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, _value: i32) {
        self.constants.insert(int.value());
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, _value: bool) {
        self.constants.insert(bool.value());
    }

    fn visit_add(&mut self, graph: &mut Rvsdg, add: Add) {
        let (lhs_src, rhs_src) = (graph.input_source(add.lhs()), graph.input_source(add.rhs()));

        if self.constants.contains(&lhs_src) && !self.constants.contains(&rhs_src) {
            tracing::debug!(
                ?add,
                ?lhs_src,
                ?rhs_src,
                "swapping add inputs {} and {}",
                add.lhs(),
                add.rhs(),
            );

            graph.remove_input_edges(add.lhs());
            graph.remove_input_edges(add.rhs());

            graph.add_value_edge(rhs_src, add.lhs());
            graph.add_value_edge(lhs_src, add.rhs());

            self.canonicalizations += 1;
            self.changed();
        }
    }

    fn visit_eq(&mut self, graph: &mut Rvsdg, eq: Eq) {
        let (lhs_src, rhs_src) = (graph.input_source(eq.lhs()), graph.input_source(eq.rhs()));

        if self.constants.contains(&lhs_src) && !self.constants.contains(&rhs_src) {
            tracing::debug!(
                ?eq,
                ?lhs_src,
                ?rhs_src,
                "swapping eq inputs {} and {}",
                eq.lhs(),
                eq.rhs(),
            );

            graph.remove_input_edges(eq.lhs());
            graph.remove_input_edges(eq.rhs());

            graph.add_value_edge(rhs_src, eq.lhs());
            graph.add_value_edge(lhs_src, eq.rhs());

            self.canonicalizations += 1;
            self.changed();
        }
    }
}
