use crate::{
    graph::{
        Add, Bool, Byte, Eq, Gamma, Input, Int, Load, Mul, Neg, Neq, Not, Output, Rvsdg, Scan,
        Store, Sub, Theta,
    },
    passes::{
        dataflow::{domain::ByteSet, Dataflow},
        utils::ChangeReport,
        Pass,
    },
    values::{Cell, Ptr},
};

impl Pass for Dataflow {
    fn pass_name(&self) -> &'static str {
        "dataflow"
    }

    fn did_change(&self) -> bool {
        self.changes.did_change()
    }

    fn reset(&mut self) {
        self.values.clear();
        self.changes.reset();
    }

    fn report(&self) -> ChangeReport {
        self.changes.as_report()
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, value: Ptr) {
        self.add_domain(int.value(), value);
    }

    fn visit_byte(&mut self, _graph: &mut Rvsdg, byte: Byte, value: Cell) {
        self.add_domain(byte.value(), value);
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, value: bool) {
        self.add_domain(bool.value(), value);
    }

    fn visit_add(&mut self, graph: &mut Rvsdg, add: Add) {
        self.compute_add(graph, add);
    }

    fn visit_sub(&mut self, graph: &mut Rvsdg, sub: Sub) {
        self.compute_sub(graph, sub);
    }

    fn visit_mul(&mut self, _graph: &mut Rvsdg, _mul: Mul) {}

    fn visit_not(&mut self, _graph: &mut Rvsdg, _not: Not) {}

    fn visit_neg(&mut self, _graph: &mut Rvsdg, _neg: Neg) {}

    fn visit_eq(&mut self, graph: &mut Rvsdg, eq: Eq) {
        self.compute_eq(graph, eq);
    }

    fn visit_neq(&mut self, graph: &mut Rvsdg, neq: Neq) {
        self.compute_neq(graph, neq);
    }

    fn visit_load(&mut self, graph: &mut Rvsdg, load: Load) {
        self.compute_load(graph, load);
    }

    fn visit_store(&mut self, graph: &mut Rvsdg, store: Store) {
        self.compute_store(graph, store);
    }

    fn visit_scan(&mut self, graph: &mut Rvsdg, scan: Scan) {
        self.compute_scan(graph, scan);
    }

    fn visit_input(&mut self, _graph: &mut Rvsdg, input: Input) {
        // Input calls are wildcards that can produce any value
        self.add_domain(input.output_value(), ByteSet::full());
    }

    fn visit_output(&mut self, _graph: &mut Rvsdg, _output: Output) {}

    fn visit_theta(&mut self, graph: &mut Rvsdg, theta: Theta) {
        self.compute_theta(graph, theta);
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, gamma: Gamma) {
        self.compute_gamma(graph, gamma);
    }
}
