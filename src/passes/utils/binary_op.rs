use crate::{
    graph::{Add, Eq, InputPort, Mul, Neq, NodeExt, OutputPort, Rvsdg, Sub},
    ir::Const,
};

pub trait BinaryOp: NodeExt {
    fn name() -> &'static str;

    fn symbol() -> &'static str;

    fn make_in_graph(graph: &mut Rvsdg, lhs: OutputPort, rhs: OutputPort) -> Self;

    fn apply(lhs: Const, rhs: Const) -> Const;

    fn lhs(&self) -> InputPort;

    fn rhs(&self) -> InputPort;

    fn value(&self) -> OutputPort;

    fn is_associative() -> bool;

    fn is_commutative() -> bool;
}

impl BinaryOp for Add {
    fn name() -> &'static str {
        "add"
    }

    fn symbol() -> &'static str {
        "+"
    }

    fn make_in_graph(graph: &mut Rvsdg, lhs: OutputPort, rhs: OutputPort) -> Self {
        graph.add(lhs, rhs)
    }

    fn apply(lhs: Const, rhs: Const) -> Const {
        lhs + rhs
    }

    fn lhs(&self) -> InputPort {
        Add::lhs(self)
    }

    fn rhs(&self) -> InputPort {
        Add::rhs(self)
    }

    fn value(&self) -> OutputPort {
        Add::value(self)
    }

    fn is_associative() -> bool {
        true
    }

    fn is_commutative() -> bool {
        true
    }
}

impl BinaryOp for Sub {
    fn name() -> &'static str {
        "sub"
    }

    fn symbol() -> &'static str {
        "-"
    }

    fn make_in_graph(graph: &mut Rvsdg, lhs: OutputPort, rhs: OutputPort) -> Self {
        graph.sub(lhs, rhs)
    }

    fn apply(lhs: Const, rhs: Const) -> Const {
        lhs - rhs
    }

    fn lhs(&self) -> InputPort {
        Sub::lhs(self)
    }

    fn rhs(&self) -> InputPort {
        Sub::rhs(self)
    }

    fn value(&self) -> OutputPort {
        Sub::value(self)
    }

    fn is_associative() -> bool {
        false
    }

    fn is_commutative() -> bool {
        false
    }
}

impl BinaryOp for Mul {
    fn name() -> &'static str {
        "mul"
    }

    fn symbol() -> &'static str {
        "*"
    }

    fn make_in_graph(graph: &mut Rvsdg, lhs: OutputPort, rhs: OutputPort) -> Self {
        graph.mul(lhs, rhs)
    }

    fn apply(lhs: Const, rhs: Const) -> Const {
        lhs * rhs
    }

    fn lhs(&self) -> InputPort {
        Mul::lhs(self)
    }

    fn rhs(&self) -> InputPort {
        Mul::rhs(self)
    }

    fn value(&self) -> OutputPort {
        Mul::value(self)
    }

    fn is_associative() -> bool {
        true
    }

    fn is_commutative() -> bool {
        true
    }
}

impl BinaryOp for Eq {
    fn name() -> &'static str {
        "eq"
    }

    fn symbol() -> &'static str {
        "=="
    }

    fn make_in_graph(graph: &mut Rvsdg, lhs: OutputPort, rhs: OutputPort) -> Self {
        graph.eq(lhs, rhs)
    }

    fn apply(lhs: Const, rhs: Const) -> Const {
        Const::Bool(lhs == rhs)
    }

    fn lhs(&self) -> InputPort {
        Eq::lhs(self)
    }

    fn rhs(&self) -> InputPort {
        Eq::rhs(self)
    }

    fn value(&self) -> OutputPort {
        Eq::value(self)
    }

    fn is_associative() -> bool {
        false
    }

    fn is_commutative() -> bool {
        true
    }
}

impl BinaryOp for Neq {
    fn name() -> &'static str {
        "neq"
    }

    fn symbol() -> &'static str {
        "!="
    }

    fn make_in_graph(graph: &mut Rvsdg, lhs: OutputPort, rhs: OutputPort) -> Self {
        graph.neq(lhs, rhs)
    }

    fn apply(lhs: Const, rhs: Const) -> Const {
        Const::Bool(lhs != rhs)
    }

    fn lhs(&self) -> InputPort {
        Neq::lhs(self)
    }

    fn rhs(&self) -> InputPort {
        Neq::rhs(self)
    }

    fn value(&self) -> OutputPort {
        Neq::value(self)
    }

    fn is_associative() -> bool {
        false
    }

    fn is_commutative() -> bool {
        true
    }
}
