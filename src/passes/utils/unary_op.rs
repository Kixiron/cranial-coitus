use crate::{
    graph::{InputPort, Neg, NodeExt, Not, OutputPort, Rvsdg},
    ir::Const,
};

pub trait UnaryOp: NodeExt {
    fn name() -> &'static str;

    fn symbol() -> &'static str;

    fn make_in_graph(graph: &mut Rvsdg, input: OutputPort) -> Self;

    fn apply(input: Const) -> Const;

    fn input(&self) -> InputPort;

    fn value(&self) -> OutputPort;
}

impl UnaryOp for Neg {
    fn name() -> &'static str {
        "neg"
    }

    fn symbol() -> &'static str {
        "-"
    }

    fn make_in_graph(graph: &mut Rvsdg, input: OutputPort) -> Self {
        graph.neg(input)
    }

    fn apply(input: Const) -> Const {
        -input
    }

    fn input(&self) -> InputPort {
        self.input()
    }

    fn value(&self) -> OutputPort {
        self.value()
    }
}

impl UnaryOp for Not {
    fn name() -> &'static str {
        "not"
    }

    fn symbol() -> &'static str {
        "!"
    }

    fn make_in_graph(graph: &mut Rvsdg, input: OutputPort) -> Self {
        graph.not(input)
    }

    fn apply(input: Const) -> Const {
        !input
    }

    fn input(&self) -> InputPort {
        self.input()
    }

    fn value(&self) -> OutputPort {
        self.value()
    }
}
