#![feature(vec_into_raw_parts)]

mod args;
mod graph;
mod ir;
mod lower_tokens;
mod parse;
mod passes;

use crate::{
    args::Args,
    graph::{EdgeKind, Rvsdg},
    ir::{IrBuilder, Pretty},
    parse::Token,
    passes::{
        AssociativeAdd, ConstDedup, ConstFolding, ConstLoads, Dce, ElimConstPhi, Pass,
        UnobservedStore,
    },
};
use clap::Clap;
use std::{
    fs::{self, File},
    path::Path,
};

// TODO: Write an evaluator so that we can actually verify optimizations
fn main() {
    set_logger();

    let args = Args::parse();
    let contents = fs::read_to_string(&args.file).expect("failed to read file");

    let dump_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/dumps");
    let _ = fs::remove_dir_all(&dump_dir);
    fs::create_dir_all(&dump_dir).unwrap();

    let span = tracing::info_span!("parsing");
    let tokens = span.in_scope(|| {
        tracing::info!("started parsing {}", args.file.display());

        let tokens = parse::parse(&contents);
        if cfg!(debug_assertions) {
            Token::debug_tokens(&tokens, File::create(dump_dir.join("tokens")).unwrap());
        }

        tokens
    });

    let mut graph = Rvsdg::new();
    let start = graph.start();

    let effect = start.effect();
    let ptr = graph.int(0).value();

    let (effect, _ptr) = lower_tokens::lower_tokens(&mut graph, ptr, effect, &tokens);
    graph.end(effect);

    graph.to_dot(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/target/dumps/lowered.dot",
    ));
    validate(&graph);

    let program = IrBuilder::new().translate(&graph);
    println!("Graph:\n{}", program.pretty_print());

    let mut pass = 1;
    loop {
        let mut changed = Dce::new().visit_graph(&mut graph);
        changed |= UnobservedStore::new().visit_graph(&mut graph);
        changed |= ConstFolding::new().visit_graph(&mut graph);
        changed |= AssociativeAdd::new().visit_graph(&mut graph);
        changed |= ElimConstPhi::new().visit_graph(&mut graph);
        changed |= ConstLoads::new(args.cells as usize).visit_graph(&mut graph);
        changed |= ConstFolding::new().visit_graph(&mut graph);
        changed |= ConstDedup::new().visit_graph(&mut graph);
        validate(&graph);

        println!("finished simplify pass #{}", pass);

        if cfg!(debug_assertions) {
            let program = IrBuilder::new().translate(&graph);
            println!("{}", program.pretty_print());
        }

        pass += 1;
        if !changed || pass >= args.iteration_limit.unwrap_or(usize::MAX) {
            break;
        }
    }

    let program = IrBuilder::new().translate(&graph);
    println!(
        "Optimized Program (took {} iterations):\n{}",
        pass,
        program.pretty_print(),
    );
}

// TODO: Turn validation into a pass
// TODO: Make validation check edge and port kinds
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
            .outputs(node_id)
            .flat_map(|(_, data)| data)
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
