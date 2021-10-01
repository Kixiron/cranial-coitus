mod args;
mod graph;
mod parse;

use crate::{
    args::Args,
    graph::{EdgeKind, Int, Node, NodeId, OutputPort, Rvsdg, ThetaData},
    parse::Token,
};
use clap::Clap;
use std::{
    collections::HashMap,
    fs::{self, File},
    mem,
    path::Path,
};

// TODO: Make IR and serialization for rvsdg, use IR for debugging since the graph
//       sucks to read
fn main() {
    let args = Args::parse();
    let contents = fs::read_to_string(&args.file).expect("failed to read file");

    let dump_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/dumps");
    let _ = fs::remove_dir_all(&dump_dir);
    fs::create_dir_all(&dump_dir).unwrap();

    let tokens = parse::parse(&contents);
    Token::debug_tokens(&tokens, File::create(dump_dir.join("tokens")).unwrap());

    let mut graph = Rvsdg::new();
    let start = graph.start();

    let effect = start.effect();
    let ptr = graph.int(0).value();

    let effect = lower_tokens(&mut graph, ptr, effect, &tokens);
    graph.end(effect);

    graph.to_dot(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/target/dumps/lowered.dot",
    ));

    println!("Graph: {:#?}", graph);

    let mut pass = 1;
    while simplify(
        &mut vec![Some(0); args.cells as usize],
        &mut HashMap::new(),
        &mut graph,
    ) {
        println!("finished simplify pass #{}", pass);
        pass += 1;
    }

    println!("Simplified Graph: {:#?}", graph);

    // let env = Env::new(args.cells as usize);
    // let (env, _) = analyze_node(&graph, env, start.id());
    //
    // println!("Env: {:#?}", env);
}

fn lower_tokens(
    graph: &mut Rvsdg,
    mut ptr: OutputPort,
    mut effect: OutputPort,
    tokens: &[Token],
) -> OutputPort {
    let (one, neg_one) = (graph.int(1).value(), graph.int(-1).value());

    for token in tokens {
        match token {
            Token::IncPtr => ptr = graph.add(ptr, one).value(),
            Token::DecPtr => ptr = graph.add(ptr, neg_one).value(),

            Token::Inc => {
                // Load the pointed-to cell's current value
                let load = graph.load(ptr, effect);
                effect = load.effect();

                // Increment the loaded cell's value
                let inc = graph.add(load.value(), one).value();

                // Store the incremented value into the pointed-to cell
                let store = graph.store(ptr, inc, effect);
                effect = store.effect();
            }
            Token::Dec => {
                // Load the pointed-to cell's current value
                let load = graph.load(ptr, effect);
                effect = load.effect();

                // Decrement the loaded cell's value
                let dec = graph.add(load.value(), neg_one).value();

                // Store the decremented value into the pointed-to cell
                let store = graph.store(ptr, dec, effect);
                effect = store.effect();
            }

            Token::Output => {
                // Load the pointed-to cell's current value
                let load = graph.load(ptr, effect);
                effect = load.effect();

                // Output the value of the loaded cell
                let output = graph.output(load.value(), effect);
                effect = output.effect();
            }
            Token::Input => {
                // Get user input
                let input = graph.input(effect);
                effect = input.effect();

                // Store the input's result to the currently pointed-to cell
                let store = graph.store(ptr, input.value(), effect);
                effect = store.effect();
            }

            Token::Loop(body) => {
                // TODO Wrap theta in a phi node that decides to run the body based on whether `load(ptr)` is zero
                graph.theta([ptr], effect, |graph, effect, inputs| {
                    let [ptr]: [OutputPort; 1] = inputs.try_into().unwrap();
                    let effect = lower_tokens(graph, ptr, effect, body);

                    let zero = graph.int(0);
                    let load = graph.load(ptr, effect);
                    let condition = graph.neq(load.value(), zero.value());

                    ThetaData::new([], condition.value(), load.effect())
                });
            }
        }
    }

    effect
}

#[derive(Debug, Default)]
struct Env {
    ptr: Option<i32>,
    tape: Vec<Option<i32>>,
    evaluated: HashMap<NodeId, Option<Value>>,
}

impl Env {
    fn new(tape_len: usize) -> Self {
        Self {
            ptr: Some(0),
            tape: vec![Some(0); tape_len],
            evaluated: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
enum Value {
    Unknown,
    Int(i32),
    Bool(bool),
    Array(Vec<Value>),
}

impl Value {
    pub fn partial_eq(&self, other: &Self) -> Option<bool> {
        match (self, other) {
            (&Self::Int(lhs), &Self::Int(rhs)) => Some(lhs == rhs),
            (&Self::Bool(lhs), &Self::Bool(rhs)) => Some(lhs == rhs),
            (Self::Array(lhs), Self::Array(rhs)) => Some(
                lhs.len() == rhs.len()
                    && lhs.iter().zip(rhs).fold(Some(true), |acc, (lhs, rhs)| {
                        Some(acc? && lhs.partial_eq(rhs)?)
                    })?,
            ),

            (Self::Unknown, _) | (_, Self::Unknown) => None,
            (_, _) => {
                println!("tried to compare {:?} == {:?}", self, other);
                None
            }
        }
    }
}

fn simplify(
    tape: &mut Vec<Option<i32>>,
    known_values: &mut HashMap<NodeId, i32>,
    graph: &mut Rvsdg,
) -> bool {
    let mut changed = false;
    let nodes: Vec<_> = graph.nodes().collect();

    for node_id in nodes {
        if let Some(node) = graph.try_node(node_id).cloned() {
            // If the current node is completely unused, cull it
            // TODO: Trace from start nodes to end nodes in order to find all live nodes
            if graph.incoming_count(node_id) == 0 && graph.outgoing_count(node_id) == 0 {
                println!("removed node {:?}", node);
                debug_assert!(!node.is_start() && !node.is_end());

                graph.remove_node(node_id);
                changed = true;
                continue;
            }

            match node {
                Node::Add(add) => {
                    if let [lhs, rhs] = *graph
                        .inputs(node_id)
                        .filter_map(|(_, operand, _)| {
                            operand
                                .as_int()
                                .map(|(_, value)| value)
                                .or_else(|| known_values.get(&operand.node_id()).copied())
                        })
                        .collect::<Vec<_>>()
                    {
                        let sum = lhs + rhs;
                        println!("evaluated add {:?} to {}", add, sum);

                        graph.remove_inputs(node_id);
                        graph.replace_node(node_id, Node::Int(Int::new(node_id, add.value()), sum));
                        known_values.insert(node_id, sum);

                        changed = true;
                    }
                }

                Node::Load(load) => {
                    let ptr = graph.inputs(node_id).find_map(|(port, ptr_node, _)| {
                        if port == load.ptr() {
                            ptr_node
                                .as_int()
                                .map(|(_, ptr)| ptr)
                                .or_else(|| known_values.get(&ptr_node.node_id()).copied())
                        } else {
                            None
                        }
                    });

                    if let Some(offset) = ptr {
                        let offset = offset.rem_euclid(tape.len() as i32) as usize;

                        if let Some(value) = tape[offset] {
                            println!("replacing {:?} with {}", load, value);

                            graph.splice_ports(load.effect_in(), load.effect());
                            graph.remove_output(load.effect());
                            graph.remove_inputs(node_id);
                            graph.replace_node(
                                node_id,
                                Node::Int(Int::new(node_id, load.value()), value),
                            );

                            changed = true;
                        }
                    }
                }

                Node::Store(store) => {
                    let effect_output = graph.get_output(store.effect());
                    if let Some((node, kind)) = effect_output {
                        debug_assert_eq!(kind, EdgeKind::Effect);

                        // If the next effect can't observe the store, remove this store
                        if node.is_end() || node.is_store() {
                            println!("removing unobserved store {:?}", store);
                            changed = true;

                            graph.splice_ports(store.effect_in(), store.effect());
                            graph.remove_node(node_id);

                            continue;
                        }

                    // If there's no consumer of this store, it's completely redundant
                    } else if effect_output.is_none() {
                        println!("removing unconsumed store {:?}", store);
                        changed = true;

                        graph.remove_node(node_id);
                        continue;
                    }

                    let ptr = graph.inputs(node_id).find_map(|(port, ptr_node, _)| {
                        if port == store.ptr() {
                            ptr_node
                                .as_int()
                                .map(|(_, ptr)| ptr)
                                .or_else(|| known_values.get(&ptr_node.node_id()).copied())
                        } else {
                            None
                        }
                    });

                    if let Some(offset) = ptr {
                        let offset = offset.rem_euclid(tape.len() as i32) as usize;

                        let stored_value =
                            graph.inputs(node_id).find_map(|(port, value_node, _)| {
                                if port == store.value() {
                                    value_node.as_int().map(|(_, value)| value).or_else(|| {
                                        known_values.get(&value_node.node_id()).copied()
                                    })
                                } else {
                                    None
                                }
                            });

                        tape[offset] = stored_value;
                        if let Some(value) = stored_value {
                            known_values.insert(node_id, value);

                            // If the load's input is known but not constant, replace
                            // it with a constant input
                            if !graph.get_input(store.value()).0.is_int() {
                                let int = graph.int(value);
                                known_values.insert(int.node(), value);

                                graph.remove_input(store.value());
                                graph.add_value_edge(int.value(), store.value());

                                println!("redirected {:?} to a constant of {}", store, value);
                                changed = true;
                            }
                        }
                    } else {
                        println!("unknown store {:?}, invalidating tape", store);

                        // Invalidate the whole tape
                        for cell in tape.iter_mut() {
                            *cell = None;
                        }
                    }
                }

                Node::Int(_, value) => {
                    let replaced = known_values.insert(node_id, value);
                    debug_assert!(replaced.is_none() || replaced == Some(value));
                }

                // FIXME: This is hacky and loses a bunch of much-needed information
                Node::Theta(mut theta) => {
                    let mut known = HashMap::new();
                    changed |= simplify(&mut vec![None; tape.len()], &mut known, theta.body_mut());

                    // Invalidate the whole tape
                    for cell in tape.iter_mut() {
                        *cell = None;
                    }

                    // Replace the old body with the simplified one
                    *graph.get_node_mut(node_id).to_theta_mut().body_mut() = theta.into_body();
                }

                Node::Array(_, _) => {}
                Node::Start(_) => {}
                Node::End(_) => {}
                Node::Input(_) => {}
                Node::Output(_) => {}
                Node::InputPort(_) => {}
                Node::OutputPort(_) => {}
                Node::Eq(_) => {}
                Node::Not(_) => {}
            }
        }
    }

    changed
}

fn analyze_node(graph: &Rvsdg, mut env: Env, node_id: NodeId) -> (Env, Option<Value>) {
    if let Some(value) = env.evaluated.get(&node_id).cloned() {
        return (env, value);
    }

    let inputs: Vec<_> = graph.inputs(node_id).collect();
    let outputs: Vec<_> = graph.outputs(node_id).collect();
    let node = graph.get_node(node_id);

    // println!(
    //     "Node: {:?}\nInputs: {:#?}\nOutputs: {:#?}",
    //     node, inputs, outputs,
    // );

    let input_values: HashMap<_, _> = inputs
        .iter()
        .map(|&(port, node, _)| {
            let (env2, value) = analyze_node(graph, mem::take(&mut env), node.node_id());
            env = env2;

            (port, value)
        })
        .collect();

    let value = match node {
        &Node::Int(_, value) => {
            debug_assert!(inputs.is_empty());
            Some(Value::Int(value))
        }

        &Node::Array(ref array, len) => {
            debug_assert_eq!(len as usize, input_values.len());
            let elements = array
                .elements()
                .iter()
                .map(|elem| input_values.get(elem).cloned().unwrap().unwrap())
                .collect();

            Some(Value::Array(elements))
        }

        Node::Add(add) => {
            let lhs = input_values[&add.lhs()].clone().unwrap();
            let rhs = input_values[&add.rhs()].clone().unwrap();

            if let (&Value::Int(lhs), &Value::Int(rhs)) = (&lhs, &rhs) {
                Some(Value::Int(lhs + rhs))
            } else {
                println!("tried to add {:?} and {:?}", lhs, rhs);
                Some(Value::Unknown)
            }
        }

        Node::Load(load) => {
            let offset = input_values[&load.ptr()].clone().unwrap();

            if let Value::Int(offset) = offset {
                // TODO: Wrap offset to tape
                match env.tape[offset as usize] {
                    Some(value) => Some(Value::Int(value)),
                    None => Some(Value::Unknown),
                }
            } else {
                println!("tried to load {:?}", offset);
                Some(Value::Unknown)
            }
        }

        Node::Store(store) => {
            let offset = input_values[&store.ptr()].clone().unwrap();
            let value = input_values[&store.value()].clone().unwrap();

            if let Value::Int(offset) = offset {
                // TODO: Wrap offset to tape
                env.tape[offset as usize] = match value {
                    Value::Int(int) => Some(int),
                    Value::Unknown => None,
                    other => {
                        println!("tried to store {:?} to {}", other, offset);
                        None
                    }
                };
            } else {
                println!("tried to store to {:?}, invalidating tape", offset);
                for cell in env.tape.iter_mut() {
                    *cell = None;
                }
            }

            None
        }

        Node::Theta(theta) => {
            let mut body_env = Env {
                ptr: env.ptr,
                tape: env.tape,
                evaluated: HashMap::new(),
            };

            // Populate the body's env with the theta's input parameters
            for (input, &param) in theta.inputs().iter().zip(theta.input_params()) {
                body_env
                    .evaluated
                    .insert(param, input_values[input].clone());
            }

            let (body_env, _value) = analyze_node(theta.body(), body_env, theta.start_node());
            env = Env {
                ptr: body_env.ptr,
                tape: body_env.tape,
                evaluated: env.evaluated,
            };

            // TODO: Output values

            None
        }

        Node::Input(_) => Some(Value::Unknown),

        Node::Start(_) | Node::End(_) | Node::Output(_) => None,

        Node::InputPort(_) | Node::OutputPort(_) => {
            println!("tried to evaluate an input or output port");
            None
        }

        Node::Eq(eq) => {
            let lhs = input_values[&eq.lhs()].clone().unwrap();
            let rhs = input_values[&eq.rhs()].clone().unwrap();

            Some(
                lhs.partial_eq(&rhs)
                    .map_or_else(|| Value::Unknown, Value::Bool),
            )
        }

        Node::Not(not) => {
            let input = input_values[&not.input()].clone().unwrap();

            Some(match input {
                Value::Bool(input) => Value::Bool(!input),
                Value::Unknown => Value::Unknown,

                other => {
                    println!("tried to use logical not on {:?}", other);
                    Value::Unknown
                }
            })
        }
    };
    env.evaluated.insert(node.node_id(), value.clone());

    for (_, data) in outputs {
        if let Some((node, _)) = data {
            let (env2, _value) = analyze_node(graph, mem::take(&mut env), node.node_id());
            env = env2;
        }
    }

    (env, value)
}
