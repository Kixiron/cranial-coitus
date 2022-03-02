use crate::{
    graph::{Add, InputPort, Mul, NodeExt, NodeId, OutputPort, Sub},
    values::Ptr,
};

// TODO: Eq?
pub trait BinOp {
    fn name() -> &'static str;

    fn symbol() -> &'static str;

    fn combine(lhs: Ptr, rhs: Ptr) -> Ptr;

    fn node(&self) -> NodeId;

    fn lhs(&self) -> InputPort;

    fn rhs(&self) -> InputPort;

    fn value(&self) -> OutputPort;

    fn is_associative() -> bool;

    fn is_commutative() -> bool;
}

impl BinOp for Add {
    fn name() -> &'static str {
        "add"
    }

    fn symbol() -> &'static str {
        "+"
    }

    fn combine(lhs: Ptr, rhs: Ptr) -> Ptr {
        lhs + rhs
    }

    fn node(&self) -> NodeId {
        NodeExt::node(self)
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

impl BinOp for Sub {
    fn name() -> &'static str {
        "sub"
    }

    fn symbol() -> &'static str {
        "-"
    }

    fn combine(lhs: Ptr, rhs: Ptr) -> Ptr {
        lhs - rhs
    }

    fn node(&self) -> NodeId {
        NodeExt::node(self)
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

impl BinOp for Mul {
    fn name() -> &'static str {
        "mul"
    }

    fn symbol() -> &'static str {
        "*"
    }

    fn combine(lhs: Ptr, rhs: Ptr) -> Ptr {
        lhs * rhs
    }

    fn node(&self) -> NodeId {
        NodeExt::node(self)
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
