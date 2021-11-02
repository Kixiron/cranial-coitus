use crate::{
    graph::{
        Add, Bool, EdgeKind, Eq, Gamma, InputPort, Int, Neg, Node, NodeExt, NodeId, Not,
        OutputPort, Rvsdg, Theta,
    },
    passes::Pass,
    utils::{AssertNone, HashMap, HashSet},
};

/// Loop invariant code motion
// TODO: Pull out expressions that only depend on invariant inputs
// TODO: Demote variant inputs to invariant ones where possible
pub struct Licm {
    changed: bool,
    within_theta: bool,
    invariant_exprs: HashSet<OutputPort>,
}

impl Licm {
    pub fn new() -> Self {
        Self {
            changed: false,
            within_theta: false,
            invariant_exprs: HashSet::with_hasher(Default::default()),
        }
    }

    fn within_theta(mut self, within_theta: bool) -> Self {
        self.within_theta = within_theta;
        self
    }

    fn changed(&mut self) {
        self.changed = true;
    }

    fn input_invariant(&self, graph: &Rvsdg, input: InputPort) -> bool {
        self.invariant_exprs.contains(&graph.input_source(input))
    }

    fn pull_out_constants(
        &mut self,
        graph: &mut Rvsdg,
        body: &mut Rvsdg,
        invariant_exprs: &HashSet<OutputPort>,
    ) -> Vec<(OutputPort, NodeId)> {
        // TODO: Buffers
        let mut invariant_exprs: Vec<_> = invariant_exprs.iter().copied().collect();
        invariant_exprs.sort_unstable();

        // TODO: Buffers
        let (mut params, mut removals, mut param_to_new) = (
            Vec::with_capacity(invariant_exprs.len()),
            HashSet::with_capacity_and_hasher(invariant_exprs.len(), Default::default()),
            HashMap::with_capacity_and_hasher(invariant_exprs.len(), Default::default()),
        );

        while let Some(port) = invariant_exprs.pop() {
            let node_id = body.port_parent(port);

            match *body.get_node(node_id) {
                Node::Int(old_int, value) => {
                    let input = body.input_param(EdgeKind::Value);
                    body.rewire_dependents(old_int.value(), input.output());
                    removals.insert(old_int.node());

                    let int = graph.int(value);
                    params.push((int.value(), input.node()));
                    param_to_new
                        .insert(input.output(), int.value())
                        .debug_unwrap_none();

                    tracing::trace!(?old_int, ?int, "pulled int {} out of subgraph", value);
                    self.changed();
                }

                Node::Bool(old_bool, value) => {
                    let input = body.input_param(EdgeKind::Value);
                    body.rewire_dependents(old_bool.value(), input.output());
                    removals.insert(old_bool.node());

                    let bool = graph.bool(value);
                    params.push((bool.value(), input.node()));
                    param_to_new
                        .insert(input.output(), bool.value())
                        .debug_unwrap_none();

                    tracing::trace!(?old_bool, ?bool, "pulled bool {} out of subgraph", value);
                    self.changed();
                }

                Node::Add(old_add) => {
                    let lhs = param_to_new.get(&body.input_source(old_add.lhs())).copied();
                    let rhs = param_to_new.get(&body.input_source(old_add.rhs())).copied();

                    if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
                        let input = body.input_param(EdgeKind::Value);
                        body.rewire_dependents(old_add.value(), input.output());
                        removals.insert(old_add.node());

                        let add = graph.add(lhs, rhs);
                        params.push((add.value(), input.node()));
                        param_to_new
                            .insert(input.output(), add.value())
                            .debug_unwrap_none();

                        tracing::trace!(?old_add, ?add, "pulled add out of subgraph");
                        self.changed();
                    } else {
                        invariant_exprs.insert(0, port);
                    }
                }

                Node::Eq(old_eq) => {
                    let lhs = param_to_new.get(&body.input_source(old_eq.lhs())).copied();
                    let rhs = param_to_new.get(&body.input_source(old_eq.rhs())).copied();

                    if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
                        let input = body.input_param(EdgeKind::Value);
                        body.rewire_dependents(old_eq.value(), input.output());
                        removals.insert(old_eq.node());

                        let eq = graph.add(lhs, rhs);
                        params.push((eq.value(), input.node()));
                        param_to_new
                            .insert(input.output(), eq.value())
                            .debug_unwrap_none();

                        tracing::trace!(?old_eq, ?eq, "pulled eq out of subgraph");
                        self.changed();
                    } else {
                        invariant_exprs.insert(0, port);
                    }
                }

                Node::Not(old_not) => {
                    let negated = param_to_new
                        .get(&body.input_source(old_not.input()))
                        .copied();

                    if let Some(negated) = negated {
                        let input = body.input_param(EdgeKind::Value);
                        body.rewire_dependents(old_not.value(), input.output());
                        removals.insert(old_not.node());

                        let not = graph.not(negated);
                        params.push((not.value(), input.node()));
                        param_to_new
                            .insert(input.output(), not.value())
                            .debug_unwrap_none();

                        tracing::trace!(?old_not, ?not, "pulled not out of subgraph");
                        self.changed();
                    } else {
                        invariant_exprs.insert(0, port);
                    }
                }

                Node::Neg(old_neg) => {
                    let negated = param_to_new
                        .get(&body.input_source(old_neg.input()))
                        .copied();

                    if let Some(negated) = negated {
                        let input = body.input_param(EdgeKind::Value);
                        body.rewire_dependents(old_neg.value(), input.output());
                        removals.insert(old_neg.node());

                        let neg = graph.neg(negated);
                        params.push((neg.value(), input.node()));
                        param_to_new
                            .insert(input.output(), neg.value())
                            .debug_unwrap_none();

                        tracing::trace!(?old_neg, ?neg, "pulled neg out of subgraph");
                        self.changed();
                    } else {
                        invariant_exprs.insert(0, port);
                    }
                }

                ref node => {
                    tracing::warn!("unhandled invariant node kind: {:?}", node);
                }
            }
        }

        body.bulk_remove_nodes(&removals);

        params
    }
}

// TODO: Invariant branches and loops
impl Pass for Licm {
    fn pass_name(&self) -> &str {
        "loop-invariant-code-motion"
    }

    fn did_change(&self) -> bool {
        self.changed
    }

    fn reset(&mut self) {
        self.changed = false;
    }

    fn visit_theta(&mut self, graph: &mut Rvsdg, mut theta: Theta) {
        let mut visitor = Self::new().within_theta(true);
        self.changed |= visitor.visit_graph(theta.body_mut());

        // Pull out any constants that are within the loop body
        let pulled_constants =
            self.pull_out_constants(graph, theta.body_mut(), &visitor.invariant_exprs);
        for (constant, param) in pulled_constants {
            let port = graph.input_port(theta.node(), EdgeKind::Value);
            graph.add_value_edge(constant, port);

            theta.add_invariant_input_raw(port, param);
        }

        graph.replace_node(theta.node(), Node::Theta(Box::new(theta)));
    }

    fn visit_gamma(&mut self, graph: &mut Rvsdg, mut gamma: Gamma) {
        let mut visitor = Self::new();

        self.changed |= visitor.visit_graph(gamma.true_mut());
        visitor.reset();
        // self.pull_out_constants(graph, gamma.true_mut());

        self.changed |= visitor.visit_graph(gamma.false_mut());
        // self.pull_out_constants(graph, gamma.false_mut());

        graph.replace_node(gamma.node(), Node::Gamma(Box::new(gamma)));
    }

    fn visit_int(&mut self, _graph: &mut Rvsdg, int: Int, _value: i32) {
        if self.within_theta {
            self.invariant_exprs.insert(int.value());
        }
    }

    fn visit_bool(&mut self, _graph: &mut Rvsdg, bool: Bool, _value: bool) {
        if self.within_theta {
            self.invariant_exprs.insert(bool.value());
        }
    }

    fn visit_add(&mut self, graph: &mut Rvsdg, add: Add) {
        if self.within_theta
            && self.input_invariant(graph, add.rhs())
            && self.input_invariant(graph, add.lhs())
        {
            self.invariant_exprs.insert(add.value());
        }
    }

    fn visit_eq(&mut self, graph: &mut Rvsdg, eq: Eq) {
        if self.within_theta
            && self.input_invariant(graph, eq.rhs())
            && self.input_invariant(graph, eq.lhs())
        {
            self.invariant_exprs.insert(eq.value());
        }
    }

    fn visit_not(&mut self, graph: &mut Rvsdg, not: Not) {
        if self.within_theta && self.input_invariant(graph, not.input()) {
            self.invariant_exprs.insert(not.value());
        }
    }

    fn visit_neg(&mut self, graph: &mut Rvsdg, neg: Neg) {
        if self.within_theta && self.input_invariant(graph, neg.input()) {
            self.invariant_exprs.insert(neg.value());
        }
    }
}

impl Default for Licm {
    fn default() -> Self {
        Self::new()
    }
}

test_opts! {
    move_constants_from_theta,
    passes = [Licm::new()],
    output = [],
    |graph, effect| {
        graph.theta([], [], effect, |graph, effect, _invariant, _variant| {
            graph.int(20);
            let bool = graph.bool(false);

            ThetaData::new([], bool.value(), effect)
        })
        .output_effect()
        .unwrap()
    },
}
