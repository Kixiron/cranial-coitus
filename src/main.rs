#![feature(vec_into_raw_parts, hash_drain_filter)]

mod args;
mod codegen;
mod graph;
mod ir;
mod lower_tokens;
mod parse;
mod passes;
mod patterns;
mod utils;

use crate::{
    args::Args,
    graph::{EdgeKind, Node, Rvsdg},
    ir::{IrBuilder, Pretty},
    parse::Token,
    passes::{
        AddSubLoop, AssociativeAdd, ConstDedup, ConstFolding, Dce, ElimConstGamma, Mem2Reg, Pass,
        UnobservedStore, ZeroLoop,
    },
};
use clap::Clap;
use similar::{Algorithm, TextDiff};
use std::{
    collections::{BTreeSet, VecDeque},
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::{Duration, Instant},
};

// TODO: Write an evaluator so that we can actually verify optimizations
// TODO: Codegen via https://docs.rs/iced-x86/1.15.0/iced_x86/index.html
fn main() {
    set_logger();

    let args = Args::parse();
    let contents = fs::read_to_string(&args.file).expect("failed to read file");

    let dump_dir = Path::new("./dumps").join(args.file.with_extension("").file_name().unwrap());

    let _ = fs::remove_dir_all(&dump_dir);
    fs::create_dir_all(&dump_dir).unwrap();

    let start_time = Instant::now();

    let span = tracing::info_span!("parsing");
    let tokens = span.in_scope(|| {
        tracing::info!("started parsing {}", args.file.display());

        let tokens = parse::parse(&contents);

        let elapsed = start_time.elapsed();
        tracing::info!("finished parsing {} in {:#?}", args.file.display(), elapsed);

        if cfg!(debug_assertions) {
            let start_time = Instant::now();

            let token_file = BufWriter::new(File::create(dump_dir.join("tokens")).unwrap());
            Token::debug_tokens(&tokens, token_file);

            let elapsed = start_time.elapsed();
            tracing::info!(
                "finished debugging tokens for {} in {:#?}",
                args.file.display(),
                elapsed,
            );
        }

        tokens
    });

    tracing::info!("started building rvsdg");
    let graph_building_start = Instant::now();

    let mut graph = Rvsdg::new();
    let start = graph.start();

    let effect = start.effect();
    let ptr = graph.int(0).value();

    let (effect, _ptr) = lower_tokens::lower_tokens(&mut graph, ptr, effect, &tokens);
    graph.end(effect);

    let elapsed = graph_building_start.elapsed();
    tracing::info!("finished building rvsdg in {:#?}", elapsed);

    validate(&graph);

    let input_stats = graph.stats();
    let input_program = IrBuilder::new().translate(&graph).pretty_print();
    fs::write(dump_dir.join("input.cir"), &input_program).unwrap();

    let mut evolution = BufWriter::new(File::create(dump_dir.join("evolution.diff")).unwrap());
    write!(&mut evolution, ">>>>> input\n{}", input_program).unwrap();

    let mut passes: Vec<Box<dyn Pass>> = vec![
        Box::new(Dce::new()),
        Box::new(UnobservedStore::new()),
        Box::new(ConstFolding::new()),
        Box::new(AssociativeAdd::new()),
        Box::new(ZeroLoop::new()),
        Box::new(Mem2Reg::new(args.cells as usize)),
        Box::new(AddSubLoop::new()),
        Box::new(Dce::new()),
        Box::new(ElimConstGamma::new()),
        Box::new(ConstFolding::new()),
        Box::new(ConstDedup::new()),
    ];
    let (mut pass_num, mut stack, mut visited, mut buffer, mut previous_graph) = (
        1,
        VecDeque::new(),
        BTreeSet::new(),
        Vec::new(),
        input_program.clone(),
    );

    loop {
        let mut changed = false;

        for (pass_idx, pass) in passes.iter_mut().enumerate() {
            let span = tracing::info_span!("pass", pass = pass.pass_name());
            span.in_scope(|| {
                tracing::info!(
                    "running {} (pass #{}.{})",
                    pass.pass_name(),
                    pass_num,
                    pass_idx,
                );

                let start_time = Instant::now();
                changed |=
                    pass.visit_graph_inner(&mut graph, &mut stack, &mut visited, &mut buffer);
                let elapsed = start_time.elapsed();

                tracing::info!(
                    "finished running {} in {:#?} (pass #{}.{}, {})",
                    pass.pass_name(),
                    elapsed,
                    pass_num,
                    pass_idx,
                    if pass.did_change() {
                        "changed"
                    } else {
                        "didn't change"
                    },
                );

                pass.reset();
                stack.clear();

                let current_graph = IrBuilder::new().translate(&graph).pretty_print();

                let diff = diff_ir(&previous_graph, &current_graph);

                if !diff.is_empty() {
                    fs::write(
                        dump_dir.join(format!(
                            "{}-{}.{}.cir",
                            pass.pass_name(),
                            pass_num,
                            pass_idx,
                        )),
                        &current_graph,
                    )
                    .unwrap();

                    write!(
                        &mut evolution,
                        ">>>>> {}-{}.{}\n{}",
                        pass.pass_name(),
                        pass_num,
                        pass_idx,
                        diff,
                    )
                    .unwrap();

                    fs::write(
                        dump_dir.join(format!(
                            "{}-{}.{}.diff",
                            pass.pass_name(),
                            pass_num,
                            pass_idx,
                        )),
                        diff,
                    )
                    .unwrap();
                }

                previous_graph = current_graph;
            });
        }

        pass_num += 1;
        if !changed || pass_num >= args.iteration_limit.unwrap_or(usize::MAX) {
            break;
        }
    }

    let elapsed = start_time.elapsed();
    let output_program = IrBuilder::new().translate(&graph).pretty_print();

    print!("{}", output_program);

    let output_stats = graph.stats();
    let difference = input_stats.difference(output_stats);
    print!(
        "Optimized Program (took {} iterations and {:#?})\n\
         Input:\n  \
           instructions : {}\n  \
           branches     : {}\n  \
           loops        : {}\n  \
           loads        : {}\n  \
           stores       : {}\n  \
           constants    : {}\n  \
           io ops       : {}\n\
         Output:\n  \
           instructions : {}\n  \
           branches     : {}\n  \
           loops        : {}\n  \
           loads        : {}\n  \
           stores       : {}\n  \
           constants    : {}\n  \
           io ops       : {}\n\
         Change:\n  \
           instructions : {:>+6.02}%\n  \
           branches     : {:>+6.02}%\n  \
           loops        : {:>+6.02}%\n  \
           loads        : {:>+6.02}%\n  \
           stores       : {:>+6.02}%\n  \
           constants    : {:>+6.02}%\n  \
           io ops       : {:>+6.02}%\n\
        ",
        pass_num,
        elapsed,
        input_stats.instructions,
        input_stats.branches,
        input_stats.loops,
        input_stats.loads,
        input_stats.stores,
        input_stats.constants,
        input_stats.io_ops,
        output_stats.instructions,
        output_stats.branches,
        output_stats.loops,
        output_stats.loads,
        output_stats.stores,
        output_stats.constants,
        output_stats.io_ops,
        difference.instructions,
        difference.branches,
        difference.loops,
        difference.loads,
        difference.stores,
        difference.constants,
        difference.io_ops,
    );

    let io_diff = diff_ir(&input_program, &output_program);
    if !io_diff.is_empty() {
        fs::write(dump_dir.join("input-output.diff"), io_diff).unwrap();
    }

    fs::write(dump_dir.join("output.cir"), output_program).unwrap();
}

fn diff_ir(old: &str, new: &str) -> String {
    let start_time = Instant::now();

    let diff = TextDiff::configure()
        .algorithm(Algorithm::Patience)
        .deadline(Instant::now() + Duration::from_secs(1))
        .diff_lines(old, new);

    let diff = format!("{}", diff.unified_diff());

    let elapsed = start_time.elapsed();
    tracing::debug!(
        target: "timings",
        "took {:#?} to diff ir",
        elapsed,
    );

    diff
}

// TODO: Turn validation into a pass
// TODO: Make validation check edge and port kinds
fn validate(graph: &Rvsdg) {
    let start_time = Instant::now();

    validate_inner(graph);

    let elapsed = start_time.elapsed();
    tracing::debug!(
        target: "timings",
        "took {:#?} to validate graph",
        elapsed,
    );
}

fn validate_inner(graph: &Rvsdg) {
    for (node_id, node) in graph
        .node_ids()
        .map(|node_id| (node_id, graph.get_node(node_id)))
    {
        if let Node::Theta(theta) = node {
            validate_inner(theta.body());
        } else if let Node::Gamma(gamma) = node {
            validate_inner(gamma.true_branch());
            validate_inner(gamma.false_branch());
        }

        let input_desc = node.input_desc();
        let (input_effects, input_values) =
            graph
                .inputs(node_id)
                .fold((0, 0), |(effect, value), (_, _, _, edge)| match edge {
                    EdgeKind::Effect => (effect + 1, value),
                    EdgeKind::Value => (effect, value + 1),
                });

        if !input_desc.effect().contains(input_effects) {
            tracing::warn!(
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
            tracing::warn!(
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
            .get_node(node_id)
            .outputs()
            .into_iter()
            .flat_map(|output| graph.get_outputs(output))
            .fold((0, 0), |(effect, value), (_, _, edge)| match edge {
                EdgeKind::Effect => (effect + 1, value),
                EdgeKind::Value => (effect, value + 1),
            });

        if !output_desc.effect().contains(output_effects) {
            tracing::warn!(
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
        //     tracing::debug!(
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

fn set_logger() {
    use atty::Stream;
    use tracing_subscriber::{
        fmt, prelude::__tracing_subscriber_SubscriberExt, util::SubscriberInitExt, EnvFilter,
    };

    let fmt_layer = fmt::layer()
        .with_target(false)
        // .with_timer(time::uptime())
        .without_time()
        // Don't use ansi codes if we're not printing to a console
        .with_ansi(atty::is(Stream::Stdout));

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
