mod args;
mod graph;
mod ir;
mod parse;
mod passes;

use crate::{
    args::Args,
    graph::{EdgeKind, Int, Node, NodeId, OutputPort, PhiData, Rvsdg, ThetaData},
    ir::{IrBuilder, Pretty},
    parse::Token,
    passes::{ConstDedup, Dce, Pass, UnobservedStore},
};
use clap::Clap;
use std::{
    collections::HashMap,
    fs::{self, File},
    path::Path,
};

// TODO: Make IR and serialization for rvsdg, use IR for debugging since the graph
//       sucks to read
fn main() {
    set_logger();

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

    let (effect, _ptr) = lower_tokens(&mut graph, ptr, effect, &tokens);
    graph.end(effect);

    graph.to_dot(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/target/dumps/lowered.dot",
    ));
    validate(&graph);

    let program = IrBuilder::new().translate(&graph);
    println!("Graph:\n{}", program.pretty_print());

    Dce::new().visit_graph(&mut graph);
    ConstDedup::new().visit_graph(&mut graph);
    UnobservedStore::new().visit_graph(&mut graph);
    validate(&graph);

    let mut pass = 1;

    #[allow(clippy::blocks_in_if_conditions)]
    while {
        let mut changed = Dce::new().visit_graph(&mut graph);
        changed |= ConstDedup::new().visit_graph(&mut graph);
        changed |= UnobservedStore::new().visit_graph(&mut graph);
        changed |= simplify(
            &mut vec![Some(0); args.cells as usize],
            &mut HashMap::new(),
            &mut graph,
            false,
        );

        changed
    } {
        Dce::new().visit_graph(&mut graph);
        ConstDedup::new().visit_graph(&mut graph);
        UnobservedStore::new().visit_graph(&mut graph);
        validate(&graph);

        let program = IrBuilder::new().translate(&graph);
        println!(
            "finished simplify pass #{}\n{}",
            pass,
            program.pretty_print(),
        );
        pass += 1;
    }

    while Dce::new().visit_graph(&mut graph) {}
    while ConstDedup::new().visit_graph(&mut graph) {}
    validate(&graph);

    let program = IrBuilder::new().translate(&graph);
    println!("Simplified Graph:\n{}", program.pretty_print());

    validate(&graph);
}

fn lower_tokens(
    graph: &mut Rvsdg,
    mut ptr: OutputPort,
    mut effect: OutputPort,
    tokens: &[Token],
) -> (OutputPort, OutputPort) {
    let (zero, one, neg_one) = (
        graph.int(0).value(),
        graph.int(1).value(),
        graph.int(-1).value(),
    );

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
                // Load the current cell's value
                let load = graph.load(ptr, effect);
                effect = load.effect();

                // Compare the cell's value to zero
                let cmp = graph.eq(load.value(), zero);

                // Create a phi node to decide whether or not to drop into the loop
                // Brainfuck loops are equivalent to this general structure:
                //
                // ```rust
                // if *ptr != 0 {
                //     do { ... } while *ptr != 0;
                // }
                // ```
                //
                // So we translate that into our node structure using a phi
                // node as the outer `if` and a theta as the inner tail controlled loop
                let phi = graph.phi(
                    [ptr],
                    effect,
                    cmp.value(),
                    // The truthy branch (`*ptr == 0`) is empty, we skip the loop entirely
                    // if the cell's value is already zero
                    |_graph, effect, inputs| {
                        let ptr = inputs[0];
                        PhiData::new([ptr], effect)
                    },
                    // The falsy branch where `*ptr != 0`, this is where we run the loop's actual body!
                    |graph, mut effect, inputs| {
                        let mut ptr = inputs[0];

                        // Create the inner theta node
                        let theta = graph.theta([ptr], effect, |graph, effect, inputs| {
                            let [ptr]: [OutputPort; 1] = inputs.try_into().unwrap();
                            let (effect, ptr) = lower_tokens(graph, ptr, effect, body);

                            let zero = graph.int(0);
                            let load = graph.load(ptr, effect);
                            let condition = graph.neq(load.value(), zero.value());

                            ThetaData::new([ptr], condition.value(), load.effect())
                        });

                        ptr = theta.outputs()[0];
                        effect = theta.effect_out();

                        PhiData::new([ptr], effect)
                    },
                );

                ptr = phi.outputs()[0];
                effect = phi.effect_out();
            }
        }
    }

    (effect, ptr)
}

fn validate(graph: &Rvsdg) {
    for (node_id, node) in graph
        .nodes()
        .map(|node_id| (node_id, graph.get_node(node_id)))
    {
        let input_desc = node.input_desc();
        let (input_effects, input_values) =
            graph
                .inputs(node_id)
                .fold((0, 0), |(effect, value), (_, _, _, edge)| match edge {
                    EdgeKind::Effect => (effect + 1, value),
                    EdgeKind::Value => (effect, value + 1),
                });

        if !input_desc.effect().contains(input_effects) {
            println!(
                "{:?} has {} input effects when {}..{} were expected: {:?}",
                node_id,
                input_effects,
                input_desc
                    .effect()
                    .min()
                    .map_or_else(String::new, |total| total.to_string()),
                input_desc
                    .effect()
                    .max()
                    .map_or_else(String::new, |total| total.to_string()),
                node,
            );
        }
        if !input_desc.value().contains(input_values) {
            println!(
                "{:?} has {} input values when {}..{} were expected: {:?}",
                node_id,
                input_values,
                input_desc
                    .value()
                    .min()
                    .map_or_else(String::new, |total| total.to_string()),
                input_desc
                    .value()
                    .max()
                    .map_or_else(String::new, |total| total.to_string()),
                node,
            );
        }

        let output_desc = node.output_desc();
        let (output_effects, _output_values) = graph
            .outputs(node_id)
            .flat_map(|(_, data)| data)
            .fold((0, 0), |(effect, value), (_, edge)| match edge {
                EdgeKind::Effect => (effect + 1, value),
                EdgeKind::Value => (effect, value + 1),
            });

        if !output_desc.effect().contains(output_effects) {
            println!(
                "{:?} has {} output effects when {}..{} were expected: {:?}",
                node_id,
                output_effects,
                output_desc
                    .effect()
                    .min()
                    .map_or_else(String::new, |total| total.to_string()),
                output_desc
                    .effect()
                    .max()
                    .map_or_else(String::new, |total| total.to_string()),
                node,
            );
        }

        // TODO: This should just be a "You missed an optimization!" note
        // if !output_desc.value().contains(output_values) {
        //     println!(
        //         "{:?} has {} output values when {}..{} were expected: {:?}",
        //         node_id,
        //         output_values,
        //         output_desc
        //             .value()
        //             .min()
        //             .map_or_else(String::new, |total| total.to_string()),
        //         output_desc
        //             .value()
        //             .max()
        //             .map_or_else(String::new, |total| total.to_string()),
        //         node,
        //     );
        // }
    }
}

// TODO:
// - Redundant loads
//   ```
//   _578 := load _577
//   call output(_578)
//   _579 := load _577
//   ```
// - Loop unrolling
// - Data flow propagation
// - Removing equivalent nodes
// - Arithmetic folding
//   ```
//   _563 := add _562, _560
//   _564 := add _563, _560
//   _565 := add _564, _560
//   _566 := add _565, _560
//   _567 := add _566, _560
//   _568 := add _567, _560
//   _569 := add _568, _560
//   _570 := add _569, _560
//   ```
// - Region ingress/egress elimination (remove unused input and output edges
//   that go into regions, including effect edges (but only when it's unused in
//   all sub-regions))
fn simplify(
    tape: &mut Vec<Option<i32>>,
    // TODO: Probably want to wrap this into a `Value` type
    known_values: &mut HashMap<NodeId, i32>,
    graph: &mut Rvsdg,
    within_loop: bool,
) -> bool {
    let mut changed = false;
    let nodes: Vec<_> = graph.nodes().collect();

    for node_id in nodes {
        if let Some(node) = graph.try_node(node_id).cloned() {
            match node {
                // TODO: Fuse addition
                Node::Add(add) => {
                    let inputs: [_; 2] = graph
                        .inputs(node_id)
                        .map(|(input, operand, _, _)| {
                            let value = operand.as_int().map(|(_, value)| value).or_else(|| {
                                known_values
                                    .get(&operand.node_id())
                                    .copied()
                                    .filter(|_| !within_loop)
                            });

                            (input, value)
                        })
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap();

                    if let [(_, Some(lhs)), (_, Some(rhs))] = inputs {
                        let sum = lhs + rhs;
                        println!("evaluated add {:?} to {}", add, sum);

                        let int = graph.int(sum);
                        known_values.insert(int.node(), sum);

                        graph.rewire_dependents(add.value(), int.value());
                        graph.remove_node(node_id);

                        changed = true;
                    } else if let [(_, Some(0)), (input, None)] | [(input, None), (_, Some(0))] =
                        inputs
                    {
                        let non_zero_value = graph.input_source(input);
                        println!(
                            "removing an addition by zero {:?} into a direct value of {:?}",
                            add, non_zero_value,
                        );

                        graph.rewire_dependents(add.value(), non_zero_value);
                        graph.remove_node(node_id);

                        changed = true;
                    }
                }

                Node::Load(load) => {
                    let ptr = graph.inputs(node_id).find_map(|(port, ptr_node, _, _)| {
                        if port == load.ptr() {
                            ptr_node.as_int().map(|(_, ptr)| ptr).or_else(|| {
                                known_values
                                    .get(&ptr_node.node_id())
                                    .copied()
                                    .filter(|_| !within_loop)
                            })
                        } else {
                            None
                        }
                    });

                    if let Some(offset) = ptr {
                        let offset = offset.rem_euclid(tape.len() as i32) as usize;

                        if let Some(value) = tape[offset] {
                            println!("replacing {:?} with {}", load, value);

                            graph.splice_ports(load.effect_in(), load.effect());
                            graph.remove_output_edge(load.effect());
                            graph.remove_inputs(node_id);
                            graph.replace_node(
                                node_id,
                                Node::Int(Int::new(node_id, load.value()), value),
                            );
                            known_values.insert(node_id, value);

                            changed = true;
                        }
                    }
                }

                Node::Store(store) => {
                    let effect_output = graph.get_output(store.effect());
                    if let Some((node, kind)) = effect_output {
                        debug_assert_eq!(kind, EdgeKind::Effect);

                        // If the next effect can't observe the store, remove this store
                        // FIXME: `.is_store()` is wrong, it's only unobservable if the next
                        //        effect is a store to the same cell
                        if (node.is_end() || node.is_store()) && !within_loop {
                            println!("removing unobserved store {:?}", store);
                            changed = true;

                            graph.splice_ports(store.effect_in(), store.effect());
                            graph.remove_node(node_id);

                            continue;
                        } else if let Node::Load(load) = node.clone() {
                            // If there's a sequence of
                            //
                            // ```
                            // store _0, _1
                            // _2 = load _0
                            // ```
                            //
                            // we want to remove the redundant load since nothing has occurred
                            // change the pointed-to value, changing this code to
                            //
                            // ```
                            // store _0, _1
                            // _2 = _1
                            // ```
                            //
                            // but we don't have any sort of passthrough node currently (may
                            // want to fix that, could be useful) we instead rewire all dependents on
                            // the redundant load (`_2`) to instead point to the known value of the cell
                            // (`_1`) transforming this code
                            //
                            // ```
                            // store _0, _1
                            // _2 = load _0
                            // _3 = add _2, int 10
                            // ```
                            //
                            // into this code
                            //
                            // ```
                            // store _0, _1
                            // _3 = add _1, int 10
                            // ```
                            if graph.get_input(load.ptr()).0.node_id()
                                == graph.get_input(store.ptr()).0.node_id()
                            {
                                println!(
                                    "replaced dependent load with value {:?} (store: {:?})",
                                    load, store,
                                );
                                changed = true;

                                graph.splice_ports(load.effect_in(), load.effect());
                                graph.rewire_dependents(
                                    load.value(),
                                    graph.input_source(store.value()),
                                );
                                graph.remove_node(load.node());
                            }
                        }

                    // If there's no consumer of this store, it's completely redundant
                    } else if effect_output.is_none() {
                        println!("removing unconsumed store {:?}", store);
                        changed = true;

                        graph.remove_node(node_id);
                        continue;
                    }

                    let ptr = graph.inputs(node_id).find_map(|(port, ptr_node, _, _)| {
                        if port == store.ptr() {
                            ptr_node.as_int().map(|(_, ptr)| ptr).or_else(|| {
                                known_values
                                    .get(&ptr_node.node_id())
                                    .copied()
                                    .filter(|_| !within_loop)
                            })
                        } else {
                            None
                        }
                    });

                    if let Some(offset) = ptr {
                        let offset = offset.rem_euclid(tape.len() as i32) as usize;

                        let stored_value =
                            graph.inputs(node_id).find_map(|(port, value_node, _, _)| {
                                if port == store.value() {
                                    value_node.as_int().map(|(_, value)| value).or_else(|| {
                                        known_values
                                            .get(&value_node.node_id())
                                            .copied()
                                            .filter(|_| !within_loop)
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

                Node::Int(..) | Node::Bool(..) => {}

                Node::Theta(mut theta) => {
                    let mut known_vals = HashMap::new();

                    changed |= simplify(
                        &mut vec![None; tape.len()],
                        &mut known_vals,
                        theta.body_mut(),
                        true,
                    );

                    // Invalidate the whole tape
                    // FIXME: This is hacky and loses a bunch of much-needed information,
                    //        we only strictly need to invalidate *touched* nodes in the ideal
                    //        case but we will have to nuke the tape (possibly in a bounded fashion?)
                    //        if there's ever an unknown pointer **stored** to (not loaded from)
                    for cell in tape.iter_mut() {
                        *cell = None;
                    }

                    // // Collect the output ports which are unused
                    // let unused_indexes: Vec<_> = theta
                    //     .outputs()
                    //     .iter()
                    //     .zip(theta.output_params())
                    //     .enumerate()
                    //     .filter_map(|(idx, (&output, &param))| {
                    //         graph.get_output(output).map(|_| (idx, output, param))
                    //     })
                    //     .enumerate()
                    //     .map(|(offset, (idx, output, param))| (idx - offset, output, param))
                    //     .collect();
                    //
                    // // Remove the unused output ports from the outer theta node as well
                    // // as removing the (transitively) unused output params from the theta
                    // // node's body
                    // for (idx, output, param) in unused_indexes {
                    //     println!(
                    //         "removing unused output port {:?} (param: {:?}) from theta",
                    //         output, param,
                    //     );
                    //
                    //     // Remove the node & port from the graph
                    //     theta.body_mut().remove_node(param);
                    //     graph.remove_output_port(output);
                    //
                    //     // Remove the node & port from the theta node itself
                    //     let removed = theta.outputs_mut().remove(idx);
                    //     debug_assert_eq!(removed, output);
                    //
                    //     let removed = theta.output_params_mut().remove(idx);
                    //     debug_assert_eq!(removed, param);
                    //
                    //     changed = true;
                    // }

                    // Replace the old theta with the simplified one
                    *graph.get_node_mut(node_id).to_theta_mut() = theta;
                }

                // TODO: Remove dead branches on const conditions
                Node::Phi(mut phi) => {
                    changed |= simplify(
                        &mut vec![None; tape.len()],
                        &mut HashMap::new(),
                        phi.truthy_mut(),
                        true,
                    );

                    changed |= simplify(
                        &mut vec![None; tape.len()],
                        &mut HashMap::new(),
                        phi.falsy_mut(),
                        true,
                    );

                    // Invalidate the whole tape
                    // FIXME: This is hacky and loses a bunch of much-needed information,
                    //        we only strictly need to invalidate *touched* nodes in the ideal
                    //        case but we will have to nuke the tape (possibly in a bounded fashion?)
                    //        if there's ever an unknown pointer **stored** to (not loaded from)
                    for cell in tape.iter_mut() {
                        *cell = None;
                    }

                    // Replace the old phi with the simplified one
                    *graph.get_node_mut(node_id).to_phi_mut() = phi;
                }

                Node::Eq(eq) => {
                    let inputs: [_; 2] = graph
                        .inputs(node_id)
                        .map(|(input, operand, _, _)| {
                            let value = operand.as_int().map(|(_, value)| value).or_else(|| {
                                known_values
                                    .get(&operand.node_id())
                                    .copied()
                                    .filter(|_| !within_loop)
                            });

                            (input, value)
                        })
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap();

                    // If both values are known we can statically evaluate the comparison
                    if let [(_, Some(lhs)), (_, Some(rhs))] = inputs {
                        println!(
                            "replaced self-equality with {} ({:?} == {:?}) {:?}",
                            lhs == rhs,
                            lhs,
                            rhs,
                            eq,
                        );

                        let true_val = graph.bool(lhs == rhs);
                        graph.rewire_dependents(eq.value(), true_val.value());
                        graph.remove_node(node_id);

                        changed = true;
                        continue;
                    }

                    // If the operands are equal this comparison will always be true
                    let [(lhs, _), (rhs, _)] = inputs;
                    if lhs == rhs {
                        println!(
                            "replaced self-equality with true ({:?} == {:?}) {:?}",
                            lhs, rhs, eq
                        );

                        let true_val = graph.bool(true);
                        graph.rewire_dependents(eq.value(), true_val.value());
                        graph.remove_node(node_id);

                        changed = true;
                        continue;
                    }
                }

                // TODO: Simplify `not` with const eval and eliminating double-negation
                Node::Not(_) => {}

                Node::Array(_, _) => {}
                Node::Start(_) => {}
                Node::End(_) => {}
                Node::Input(_) => {}
                Node::Output(_) => {}
                Node::InputPort(_) => {}
                Node::OutputPort(_) => {}
            }
        }
    }

    changed
}

fn set_logger() {
    use tracing_subscriber::{
        fmt::{self, time},
        prelude::__tracing_subscriber_SubscriberExt,
        util::SubscriberInitExt,
        EnvFilter,
    };

    let fmt_layer = fmt::layer().with_target(false).with_timer(time::uptime());
    let filter_layer = EnvFilter::try_from_env("COITUS_LOG")
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();

    let registry = tracing_subscriber::registry().with(filter_layer);
    if cfg!(test) {
        // Use a logger that'll be captured by libtest if we're running
        // under a test harness
        registry.with(fmt_layer.with_test_writer()).init()
    } else {
        registry.with(fmt_layer).init()
    }
}
