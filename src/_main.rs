mod args;
mod graph;
mod parse;

use crate::{
    args::Args,
    graph::{Add, Array, Edge, End, Eq, Load, Node, NodeId, Read, Rvsdg, Start, Store, Write},
    parse::Token,
};
use clap::Clap;
use std::{
    fs::{self, File},
    path::Path,
};

fn main() {
    let args = Args::parse();
    let contents = fs::read_to_string(&args.file).expect("failed to read file");

    let dump_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/dumps");
    let _ = fs::remove_dir_all(&dump_dir);
    fs::create_dir_all(&dump_dir).unwrap();

    let tokens = parse::parse(&contents);
    Token::debug_tokens(&tokens, File::create(dump_dir.join("tokens")).unwrap());

    let mut graph = Rvsdg::new();

    let start = graph.add_node(Start);
    let mut last_effect = start;
    let (zero, one, neg_one) = (graph.add_node(0), graph.add_node(1), graph.add_node(-1));

    // The instruction pointer
    let mut ptr = zero;

    // Create the program tape
    // TODO: Need to incorporate this into access operations so that we can actually use the values
    // as well as modify the array's contents
    let tape = graph.add_node(Array::new(vec![zero; args.cells as usize]));

    build_graph(
        &mut graph,
        tokens,
        &mut ptr,
        &mut last_effect,
        zero,
        one,
        neg_one,
    );

    let mut tape = vec![Some(0); args.cells as usize];
    let mut ptr = Some(0);

    let end = graph.add_node(End::new(last_effect));
    graph.add_edge(last_effect, end, Edge::Effect);

    let start_inputs = graph.inputs(start);
    assert!(start_inputs.is_empty());

    let start_outputs = graph.outputs(start);
    assert_eq!(start_outputs.len(), 1);

    let (first_effect, edge) = start_outputs[0];
    assert_eq!(edge, Edge::Effect);

    for (id, node) in graph.nodes() {
        let value = evaluate_node(&graph, &mut tape, &mut ptr, id);

        println!("{:?} = {:?}", node, value);
    }
}

#[derive(Debug)]
enum Value {
    Byte(u8),
    Int(i32),
    Bool(bool),
    Unknown,
}

fn evaluate_node(
    graph: &Rvsdg,
    tape: &mut Vec<Option<u8>>,
    inst_ptr: &mut Option<usize>,
    node: NodeId,
) -> Value {
    match graph.node(node) {
        &Node::Load(Load { ptr, effect }) => {
            let ptr = evaluate_node(graph, tape, inst_ptr, ptr);

            if let Value::Int(ptr) = ptr {
                // TODO: Probably want to transpose the pointer into our current address space
                let loaded = match tape[ptr as usize] {
                    Some(cell) => Value::Byte(cell),
                    None => Value::Unknown,
                };
                println!("loaded {:?}", loaded);

                loaded
            } else {
                println!("tried to load pointer {:?}", ptr);
                Value::Unknown
            }
        }

        &Node::Store(Store { ptr, value, effect }) => {
            let ptr = evaluate_node(graph, tape, inst_ptr, ptr);
            let value = evaluate_node(graph, tape, inst_ptr, value);

            let ptr = if let Value::Int(ptr) = ptr {
                // TODO: Probably want to transpose the pointer into our current address space
                ptr as usize
            } else {
                println!("tried to store to pointer {:?}", ptr);
                return Value::Unknown;
            };

            let value = match value {
                Value::Byte(byte) => Some(byte),
                // TODO: Probably want to transpose the integer into our current address space
                Value::Int(int) => Some(int as u8),
                Value::Bool(b) => Some(b as u8),
                Value::Unknown => None,
            };

            println!("stored {:?} to {}", value, ptr);
            tape[ptr] = value;

            Value::Unknown
        }

        // Reads put an unknown value into a cell
        Node::Read(Read { effect }) => {
            if let Some(ptr) = *inst_ptr {
                tape[ptr] = None;
            }

            Value::Unknown
        }

        // Writes don't do anything we can emulate
        Node::Write(_) => Value::Unknown,

        &Node::Add(Add { lhs, rhs }) => {
            let lhs = evaluate_node(graph, tape, inst_ptr, lhs);
            let rhs = evaluate_node(graph, tape, inst_ptr, rhs);

            match (lhs, rhs) {
                (Value::Byte(lhs), Value::Byte(rhs)) => Value::Byte(lhs.wrapping_add(rhs)),
                (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs.wrapping_add(rhs)),

                (lhs, rhs) => {
                    println!("tried to add {:?} + {:?}", lhs, rhs);
                    Value::Unknown
                }
            }
        }

        Node::Theta(_) => {
            println!("attempted to evaluate theta node");
            Value::Unknown
        }

        &Node::Int(int) => Value::Int(int),

        &Node::Bool(b) => Value::Bool(b),

        Node::Array(_) => Value::Unknown,

        Node::Start(_) => Value::Unknown,

        Node::End(_) => Value::Unknown,

        &Node::Eq(Eq { lhs, rhs }) => {
            let lhs = evaluate_node(graph, tape, inst_ptr, lhs);
            let rhs = evaluate_node(graph, tape, inst_ptr, rhs);

            match (lhs, rhs) {
                (Value::Byte(lhs), Value::Byte(rhs)) => Value::Bool(lhs == rhs),
                (Value::Int(lhs), Value::Int(rhs)) => Value::Bool(lhs == rhs),
                (Value::Bool(lhs), Value::Bool(rhs)) => Value::Bool(lhs == rhs),

                (lhs, rhs) => {
                    println!("tried to equate {:?} == {:?}", lhs, rhs);
                    Value::Unknown
                }
            }
        }

        Node::Phi(_) => {
            println!("attempted to evaluate phi node");
            Value::Unknown
        }
    }
}

fn build_graph(
    graph: &mut Rvsdg,
    tokens: Vec<Token>,
    ptr: &mut NodeId,
    last_effect: &mut NodeId,
    zero: NodeId,
    one: NodeId,
    neg_one: NodeId,
) {
    for token in tokens {
        match token {
            // Offsetting the data pointer
            Token::IncPtr => *ptr = graph.add(*ptr, one),
            Token::DecPtr => *ptr = graph.add(*ptr, neg_one),

            // Incrementing & decrementing the cell at the current data pointer
            Token::Inc => {
                // Load the current value of the pointed-to cell
                let cell = graph.load(last_effect, *ptr);

                // Add one to the current cell's value
                let cell_plus_one = graph.add(cell, one);

                // Store the value back at the cell's position
                graph.store(last_effect, *ptr, cell_plus_one);
            }
            Token::Dec => {
                // Load the current value of the pointed-to cell
                let cell = graph.load(last_effect, *ptr);

                // Subtract one from the current cell's value
                let cell_minus_one = graph.add(cell, neg_one);

                // Store the value back at the cell's position
                graph.store(last_effect, *ptr, cell_minus_one);
            }

            // Input & output operations
            Token::Output => {
                // Load the current value of the pointed-to cell
                let cell = graph.load(last_effect, *ptr);

                // Write the current cell's value to the output
                graph.write(last_effect, cell);
            }
            Token::Input => {
                // Read a value from the input
                let input = graph.read(last_effect);

                // Write the input value to the current cell
                graph.store(last_effect, *ptr, input);
            }

            // Loops
            Token::Loop(body) => {
                let cell = graph.load(last_effect, *ptr);
                let cell_is_zero = graph.eq(cell, zero);

                // Brainfuck's loops are basically while loops, which are head controlled.
                // This means that to properly model their behavior with our tail controlled
                // loops, the loop body needs to be wrapped within a phi node where the entire
                // loop is skipped if the current cell is zero at the start of the loop
                graph.phi(
                    last_effect,
                    cell_is_zero,
                    |_last_effect, _graph| {},
                    |last_effect, graph| {
                        graph.theta(last_effect, |last_effect, graph| {
                            build_graph(graph, body, ptr, last_effect, zero, one, neg_one);

                            // Theta is a tail-controlled loop, we're checking if the current cell is zero
                            let cell = graph.load(last_effect, *ptr);
                            graph.eq(cell, zero)
                        });
                    },
                );
            }
        }
    }
}
