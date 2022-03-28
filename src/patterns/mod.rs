mod sexpr;

use crate::{
    graph::{Node, NodeExt, NodeId, Rvsdg},
    utils::HashMap,
};
use sexpr::Sexpr;

#[test]
fn sexpr_test() {
    let rewrite = dbg!(Rewrite::new("add-zero", "(add ?a 0)", "?a"));

    let mut graph = Rvsdg::new();
    let lhs = graph.byte(1u8).value();
    let rhs = graph.byte(0u8).value();
    let add = graph.add(lhs, rhs).into();

    rewrite.apply(&mut graph, &add);
}

#[derive(Debug)]
pub struct Rewrite<'a> {
    name: &'a str,
    pattern: Pattern<'a>,
    output: Pattern<'a>,
}

impl<'a> Rewrite<'a> {
    pub fn new(name: &'a str, pattern: &'a str, output: &'a str) -> Self {
        let pattern = Pattern::from_sexpr(&Sexpr::parse(pattern));
        let output = Pattern::from_sexpr(&Sexpr::parse(output));

        Self {
            name,
            pattern,
            output,
        }
    }

    pub fn apply(&self, graph: &mut Rvsdg, node: &Node) {
        let mut bindings = HashMap::default();
        if match_pattern(&self.pattern, graph, node, &mut bindings) {
            println!("{:?}", bindings);
        }
    }
}

fn match_pattern<'a>(
    pattern: &Pattern<'a>,
    graph: &Rvsdg,
    node: &Node,
    bindings: &mut HashMap<&'a str, NodeId>,
) -> bool {
    match pattern {
        &Pattern::Binding(binding) => {
            bindings.insert(binding, node.node());
            true
        }

        Pattern::Add(lhs_pat, rhs_pat) => {
            if let Some(add) = node.as_add() {
                let (lhs, rhs) = (add.lhs(), add.rhs());

                let lhs_node = graph.input_source_node(lhs);
                if !match_pattern(lhs_pat, graph, lhs_node, bindings) {
                    return false;
                }

                let rhs_node = graph.input_source_node(rhs);
                match_pattern(rhs_pat, graph, rhs_node, bindings)
            } else {
                false
            }
        }

        Pattern::ConstAdd(_, _) => todo!(),

        &Pattern::Int(expected) => node
            .as_int_value()
            .map(|ptr| ptr.value() == expected)
            .or_else(|| {
                node.as_byte_value()
                    // TODO: Convert to Ptr to wrap byte into pointer space
                    .map(|byte| byte.into_inner() as u16 == expected)
            })
            .unwrap_or(false),

        &Pattern::Bool(expected) => node
            .as_bool_value()
            .map(|bool| bool == expected)
            .unwrap_or(false),

        Pattern::WildCard => true,
    }
}

#[derive(Debug)]
pub enum Pattern<'a> {
    Binding(&'a str),
    Add(Box<Self>, Box<Self>),
    ConstAdd(Box<Self>, Box<Self>),
    Int(u16),
    Bool(bool),
    WildCard,
}

impl<'a> Pattern<'a> {
    pub(crate) fn from_sexpr(sexpr: &Sexpr<'a>) -> Self {
        match sexpr {
            &Sexpr::Atom(atom) => {
                if atom == "_" {
                    Self::WildCard
                } else if atom.starts_with('?') {
                    Self::Binding(atom)
                } else if matches!(atom, "true" | "false") {
                    Self::Bool(atom.parse().unwrap())
                } else {
                    Self::Int(atom.parse().unwrap())
                }
            }

            Sexpr::Cons(cons) => match cons[0].to_atom() {
                "add" => Self::Add(
                    Box::new(Self::from_sexpr(&cons[1])),
                    Box::new(Self::from_sexpr(&cons[2])),
                ),

                "+" => Self::ConstAdd(
                    Box::new(Self::from_sexpr(&cons[1])),
                    Box::new(Self::from_sexpr(&cons[2])),
                ),

                pat => panic!("unrecognized pattern: {:?}", pat),
            },
        }
    }
}
