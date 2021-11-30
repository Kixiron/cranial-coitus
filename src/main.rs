#![feature(
    asm,
    try_blocks,
    drain_filter,
    array_windows,
    result_cloned,
    slice_ptr_get,
    slice_ptr_len,
    bool_to_option,
    hash_drain_filter,
    vec_into_raw_parts,
    destructuring_assignment,
    nonnull_slice_from_raw_parts
)]

#[macro_use]
mod utils;
mod args;
mod driver;
mod graph;
mod interpreter;
mod ir;
mod jit;
mod lattice;
mod lower_tokens;
mod parse;
mod passes;
mod patterns;
mod tests;
mod union_find;

use crate::{
    args::{Args, Command},
    graph::{EdgeKind, Node, NodeExt, Rvsdg},
    interpreter::EvaluationError,
    ir::{IrBuilder, Pretty, PrettyConfig},
    jit::Jit,
    utils::{HashSet, PerfEvent},
};
use anyhow::{Context, Result};
use clap::Parser;
use std::{
    collections::VecDeque,
    fmt::Write,
    fs::{self, File},
    io::{BufWriter, Write as _},
    path::Path,
    time::Instant,
};

// TODO: Write an evaluator so that we can actually verify optimizations
// TODO: Codegen via https://docs.rs/iced-x86/1.15.0/iced_x86/index.html
// TODO: Generate ELK text files
//       https://rtsys.informatik.uni-kiel.de/elklive/elkgraph.html
//       https://github.com/eclipse/elk/pull/106
//       https://rtsys.informatik.uni-kiel.de/elklive/examples.html
//       https://rtsys.informatik.uni-kiel.de/elklive/elkgraph.html?compressedContent=IYGw5g9gTglgLgCwLYC4AEJgE8CmUcAmAUEQHYQE5qI5zBoDeRaLaOIA1gHQEz4DGcGBFLoAIgHkA6gDlmrAA7Q4AYREBnOFGAxScdegBiAJQCip+S3KUAMsABG7dVwWZ+OJDj3oARAEkZAGU-MVM0ADUAfQAVCQAFNAAJSJtTQ2ifSzQlKDgAQRAYMFJPPR4cADNgAFcQOHE-QOjjPwAhAFVo0zEs3XUYSkD2CpsICAVnYEEYADdgOBx0LWqcElYMB3Y0HwBxYCQkYEysnLg0XQVquABGRiz1gD1Trn7KdBkJY2jErIBfE+U51IlzgACY7utHs9Xos0B8vj91v91qc0Jp5jhIroIZCWE9lC8BrD4d8-mtWNYqMACAQcbjKXZHCBnK4ph4vPVtgFgqEIpEVKYZF1jEl+YLhZlcSxTgUiiUOeUqrVOWJGs02p1uuTcZgmds8jTjlLsoCQAh1HTjWh8blCW84Z9SVbWLqtj4zepJVLkVLUVBzZbjTa4HbiY7Ec6NnqfP7PfdWP949LAXMQCtA1Lg6H0IEJJ0I7ifSxEyjARAriDbkwg9CiTm806EwDcmi6AtIuWztXM7X7bn82SsoQwFQaHQuBcrrcALQAPjQ1IIXA9Q4II+oCFowAnwKu4LnC5pXFjq-Xi64qfTB7H287laIieHo8347vU7Q15f28nYKIT43W4vG2mLYp+gHou2naWn0RJDCAIxjBMXBYBASxQCsD5EEAA
fn main() -> Result<()> {
    utils::set_logger();

    let start_time = Instant::now();

    let args = Args::parse();
    match &args.command {
        Command::Run { file, no_opt } => run(&args, file, *no_opt, start_time),
        &Command::Debug {
            ref file,
            only_final_run,
            no_step_limit,
        } => debug(&args, file, only_final_run, no_step_limit, start_time),
        Command::Debugger { file } => driver::debugger(&args, file, start_time),
    }
}

fn debug(
    args: &Args,
    file: &Path,
    only_final_run: bool,
    no_step_limit: bool,
    start_time: Instant,
) -> Result<()> {
    let cells = args.cells as usize;
    let step_limit = if no_step_limit {
        usize::MAX
    } else {
        args.step_limit.unwrap_or(300_000)
    };

    let source = fs::read_to_string(file).expect("failed to read file");
    let dump_dir = Path::new("./dumps").join(file.with_extension("").file_name().unwrap());

    driver::create_dump_dir(&dump_dir)?;

    // Parse the input program and turn it into a graph
    let tokens = driver::parse_source(file, &source);
    let mut graph = driver::build_graph(&tokens);

    // Sequentialize the input program
    let (mut input_program, input_program_ir) = driver::sequentialize_graph(
        args,
        &graph,
        Some(&dump_dir.join("input.cir")),
        PrettyConfig::minimal(),
    )?;
    let input_graph_stats = graph.stats();

    // Validate the initial program
    // Note: This happens *after* we print out the initial graph for better debugging
    validate(&graph);

    let unoptimized_execution = if !only_final_run {
        let result_path = dump_dir.join("input-result.txt");
        let mut result_file = File::create(&result_path)
            .with_context(|| format!("failed to create '{}'", result_path.display()))?;

        // FIXME: Allow user supplied input
        let mut input = vec_deque![b'1', b'0'];
        let input_vec: Vec<_> = input.iter().copied().collect();
        let input = driver::array_input(&mut input);

        let mut output_vec = Vec::new();
        let output = driver::array_output(&mut output_vec);

        let (result, tape, stats, execution_duration) =
            driver::execute(step_limit, cells, input, output, false, &mut input_program);

        // FIXME: Utility function
        let input_str = String::from_utf8_lossy(&input_vec);
        writeln!(
            &mut result_file,
            "----- Input -----\n{:?}\n-----\n{:?}\n-----\n{}",
            input_vec, input_str, input_str,
        )?;

        // FIXME: Utility function
        let output_str = String::from_utf8_lossy(&output_vec);
        writeln!(
            &mut result_file,
            "----- Output -----\n{:?}\n-----\n{:?}\n-----\n{}",
            output_vec, output_str, output_str,
        )?;

        let tape_chars =
            utils::debug_collapse(&String::from_utf8_lossy(&tape).chars().collect::<Vec<_>>());
        let tape = utils::debug_collapse(&tape);

        writeln!(
            &mut result_file,
            "----- Tape -----\n{:?}\n-----\n{:?}",
            tape, tape_chars,
        )?;

        match result {
            Ok(()) => {
                println!(
                    "Unoptimized program finished execution in {:#?}\n\
                     Execution stats:\n{}\
                     output (bytes): {:?}\n\
                     output (escaped): {:?}\n\
                     output (utf8):\n{}",
                    execution_duration,
                    stats.display(),
                    output_vec,
                    output_str,
                    output_str,
                );
            }

            Err(_) => {
                println!(
                    "Unoptimized program hit the step limit of {} steps in {:#?}\n\
                     Execution stats:\n{}\
                     output (bytes): {:?}\n\
                     output (escaped): {:?}\n\
                     output (utf8): {}",
                    step_limit,
                    execution_duration,
                    stats.display(),
                    output_vec,
                    output_str,
                    output_str,
                );
            }
        }

        Some((stats, execution_duration))
    } else {
        None
    };

    let mut evolution = BufWriter::new(File::create(dump_dir.join("evolution.diff")).unwrap());
    write!(&mut evolution, ">>>>> input\n{}", input_program_ir).unwrap();

    let mut passes = passes::default_passes(args.cells as usize);
    let (mut pass_num, mut stack, mut visited, mut buffer, mut previous_program_ir) = (
        1,
        VecDeque::new(),
        HashSet::with_hasher(Default::default()),
        Vec::new(),
        input_program_ir.clone(),
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

                let event = PerfEvent::new(pass.pass_name());
                changed |=
                    pass.visit_graph_inner(&mut graph, &mut stack, &mut visited, &mut buffer);
                let elapsed = event.finish();

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

                let (_output_program, output_program_ir) =
                    driver::sequentialize_graph(args, &graph, None, PrettyConfig::minimal())?;

                let diff = utils::diff_ir(&previous_program_ir, &output_program_ir);

                if !diff.is_empty() {
                    let input_file = dump_dir.join(format!(
                        "{}-{}.{}.input.cir",
                        pass.pass_name(),
                        pass_num,
                        pass_idx,
                    ));
                    fs::write(input_file, &previous_program_ir).unwrap();

                    let output_file = dump_dir.join(format!(
                        "{}-{}.{}.cir",
                        pass.pass_name(),
                        pass_num,
                        pass_idx,
                    ));
                    fs::write(output_file, &output_program_ir).unwrap();

                    write!(
                        &mut evolution,
                        ">>>>> {}-{}.{}\n{}",
                        pass.pass_name(),
                        pass_num,
                        pass_idx,
                        diff,
                    )
                    .unwrap();

                    let diff_file = dump_dir.join(format!(
                        "{}-{}.{}.diff",
                        pass.pass_name(),
                        pass_num,
                        pass_idx,
                    ));
                    fs::write(diff_file, diff).unwrap();
                }

                previous_program_ir = output_program_ir;

                Result::<()>::Ok(())
            })?;
        }

        pass_num += 1;
        if !changed || pass_num >= args.iteration_limit.unwrap_or(usize::MAX) {
            break;
        }
    }

    let elapsed = start_time.elapsed();

    {
        let mut reports = Vec::with_capacity(passes.len());
        let mut max_len = 0;

        for pass in &passes {
            let mut pass_name = pass.pass_name().to_owned();

            let mut idx = 1;
            while reports.iter().any(|(name, _)| name == &pass_name) {
                pass_name = format!("{}-{}", pass.pass_name(), idx);
                idx += 1;
            }

            let mut report: Vec<_> = pass.report().into_iter().collect();
            report.sort_unstable_by_key(|&(aspect, _)| aspect);

            if let Some(longest) = report.iter().map(|(aspect, _)| aspect.len()).max() {
                if longest > max_len {
                    max_len = longest;
                }
            }

            reports.push((pass_name, report));
        }

        let mut report_file = BufWriter::new(
            File::create(dump_dir.join("pass-report.txt"))
                .context("failed to create report file")?,
        );
        for (pass_name, report) in &reports {
            if !report.is_empty() {
                writeln!(&mut report_file, "{}", pass_name)?;

                for (aspect, total) in report {
                    writeln!(
                        &mut report_file,
                        "{:<padding$} : {}",
                        aspect,
                        total,
                        padding = max_len,
                    )?;
                }

                writeln!(&mut report_file)?;
            }
        }

        let no_info = reports.iter().filter(|(_, report)| report.is_empty());
        if no_info.clone().count() != 0 {
            writeln!(&mut report_file, "No info for the following passes:")?;

            for (pass, _) in no_info {
                writeln!(&mut report_file, "  - {}", pass)?;
            }
        }

        report_file.flush().context("failed to flush report file")?;
    }

    let output_graph_stats = graph.stats();
    let (mut output_program, output_program_ir) = driver::sequentialize_graph(
        args,
        &graph,
        Some(&dump_dir.join("output.cir")),
        PrettyConfig::instrumented(output_graph_stats.instructions),
    )?;

    if args.print_output_ir {
        print!("{}", output_program_ir);
    }

    let io_diff = utils::diff_ir(&input_program_ir, &output_program_ir);
    if !io_diff.is_empty() {
        fs::write(dump_dir.join("input-output.diff"), io_diff).unwrap();
    }

    let result_path = dump_dir.join("output-result.txt");
    let mut result_file = File::create(&result_path)
        .with_context(|| format!("failed to create '{}'", result_path.display()))?;

    let (optimized_stats, optimized_execution_duration) = {
        // FIXME: Allow user supplied input
        let mut input = vec_deque![b'1', b'0'];
        let input_vec: Vec<_> = input.iter().copied().collect();
        let input = driver::array_input(&mut input);

        let mut output_vec = Vec::new();
        let output = driver::array_output(&mut output_vec);

        let (result, tape, stats, execution_duration) =
            driver::execute(step_limit, cells, input, output, true, &mut output_program);

        let input_str = String::from_utf8_lossy(&input_vec);
        writeln!(
            &mut result_file,
            "----- Input -----\n{:?}\n-----\n{:?}\n-----\n{}",
            input_vec, input_str, input_str,
        )?;

        // FIXME: Utility function
        let output_str = String::from_utf8_lossy(&output_vec);
        writeln!(
            &mut result_file,
            "----- Output -----\n{:?}\n-----\n{:?}\n-----\n{}",
            output_vec, output_str, output_str,
        )?;

        let tape_chars =
            utils::debug_collapse(&String::from_utf8_lossy(&tape).chars().collect::<Vec<_>>());
        let tape = utils::debug_collapse(&tape);

        writeln!(
            &mut result_file,
            "----- Tape -----\n{:?}\n-----\n{:?}",
            tape, tape_chars,
        )?;

        match result {
            Ok(()) => {
                println!(
                    "Optimized program finished execution in {:#?}\n\
                     output (bytes): {:?}\n\
                     output (escaped): {:?}\n\
                     output (utf8):\n{}",
                    execution_duration, output_vec, output_str, output_str,
                );
            }

            Err(_) => {
                println!(
                    "Optimized program hit the step limit of {} steps in {:#?}\n\
                     output (bytes): {:?}\n\
                     output (escaped): {:?}\n\
                     output (utf8): {}",
                    step_limit, execution_duration, output_vec, output_str, output_str,
                );
            }
        }

        (stats, execution_duration)
    };

    // FIXME: Utility function
    let difference = input_graph_stats.difference(output_graph_stats);
    let mut change = format!(
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
           io ops       : {:>+6.02}%\n\n",
        pass_num,
        elapsed,
        input_graph_stats.instructions,
        input_graph_stats.branches,
        input_graph_stats.loops,
        input_graph_stats.loads,
        input_graph_stats.stores,
        input_graph_stats.constants,
        input_graph_stats.io_ops,
        output_graph_stats.instructions,
        output_graph_stats.branches,
        output_graph_stats.loops,
        output_graph_stats.loads,
        output_graph_stats.stores,
        output_graph_stats.constants,
        output_graph_stats.io_ops,
        difference.instructions,
        difference.branches,
        difference.loops,
        difference.loads,
        difference.stores,
        difference.constants,
        difference.io_ops,
    );

    if let Some((unoptimized_stats, unoptimized_execution_duration)) = unoptimized_execution {
        writeln!(
            &mut change,
            "Finished unoptimized execution in {:#?}\n\
            Unoptimized execution stats:\n{}",
            unoptimized_execution_duration,
            unoptimized_stats.display(),
        )
        .unwrap();
    }

    writeln!(
        &mut change,
        "Finished optimized execution in {:#?}\n\
        Optimized execution stats:\n{}",
        optimized_execution_duration,
        optimized_stats.display(),
    )
    .unwrap();

    print!("{}", change);
    fs::write(dump_dir.join("change.txt"), change).unwrap();

    let annotated_program =
        output_program.pretty_print(PrettyConfig::instrumented(optimized_stats.instructions));
    fs::write(dump_dir.join("annotated_output.cir"), annotated_program).unwrap();

    Ok(())
}

fn run(args: &Args, file: &Path, no_opt: bool, start_time: Instant) -> Result<()> {
    let cells = args.cells as usize;
    let step_limit = args.step_limit.unwrap_or(usize::MAX);

    let contents = fs::read_to_string(file).expect("failed to read file");

    let mut graph = {
        let span = tracing::info_span!("parsing");
        let tokens = span.in_scope(|| {
            tracing::info!("started parsing {}", file.display());

            let tokens = parse::parse(&contents);

            let elapsed = start_time.elapsed();
            tracing::info!("finished parsing {} in {:#?}", file.display(), elapsed);

            tokens
        });

        tracing::info!("started building rvsdg");
        let graph_building_start = Instant::now();

        let mut graph = Rvsdg::new();
        let start = graph.start();

        let effect = start.effect();
        let ptr = graph.int(0).value();

        let (_ptr, effect) = lower_tokens::lower_tokens(&mut graph, ptr, effect, &tokens);
        graph.end(effect);

        let elapsed = graph_building_start.elapsed();
        tracing::info!("finished building rvsdg in {:#?}", elapsed);

        graph
    };

    // let ir = IrBuilder::new(false).translate(&graph);
    // println!("{}", ir.pretty_print(PrettyConfig::minimal()));
    // fs::write(
    //     concat!(env!("CARGO_MANIFEST_DIR"), "/output.asm"),
    //     Jit::new(cells).unwrap().assemble(&ir).unwrap(),
    // )?;

    {
        let mut graph = Rvsdg::new();
        let start = graph.start();
        let zero = graph.int(0);

        // Inflate the stack some to force spilling
        let mut last_effect = start.effect();
        for _ in 0..20 {
            last_effect = graph.load(zero.value(), last_effect).output_effect();
        }

        let input = graph.input(last_effect);
        let store = graph.store(zero.value(), input.output_value(), input.output_effect());
        let load = graph.load(zero.value(), store.output_effect());
        let output = graph.output(load.output_value(), load.output_effect());
        graph.end(output.output_effect());

        let ir = IrBuilder::new(false).translate(&graph);
        println!("{}", ir.pretty_print(PrettyConfig::minimal()));

        fs::write(
            concat!(env!("CARGO_MANIFEST_DIR"), "/output.asm"),
            Jit::new(cells).unwrap().assemble(&ir).unwrap(),
        )?;

        return Ok(());
    }

    let input_stats = graph.stats();
    validate(&graph);

    let opt_iters = if no_opt {
        0
    } else {
        driver::run_opt_passes(
            &mut graph,
            cells,
            args.iteration_limit.unwrap_or(usize::MAX),
        )
    };

    let mut program = IrBuilder::new(args.inline_constants).translate(&graph);
    let elapsed = start_time.elapsed();

    let output_stats = graph.stats();
    let difference = input_stats.difference(output_stats);

    // FIXME: Utility function
    println!(
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
           io ops       : {:>+6.02}%\
        ",
        opt_iters,
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

    let input = driver::stdin_input();
    let output = driver::stdout_output();
    let (result, _, stats, execution_duration) =
        driver::execute(step_limit, cells, input, output, true, &mut program);

    // FIXME: Utility function
    match result {
        Ok(()) => {
            println!(
                "\n\
                Finished execution in {:#?}\n\
                Execution stats:\n{}",
                execution_duration,
                stats.display(),
            );
        }

        Err(eval_error) => {
            debug_assert_eq!(eval_error, EvaluationError::StepLimitReached);

            println!(
                "\n\
                Program hit step limit of {} in {:#?}\n\
                Execution stats:\n{}",
                step_limit,
                execution_duration,
                stats.display(),
            );
        }
    }

    Ok(())
}

// TODO: Turn validation into a pass
// TODO: Make validation check edge and port kinds
fn validate(graph: &Rvsdg) {
    tracing::debug!(
        target: "timings",
        "started validating graph",
    );
    let start_time = Instant::now();

    let mut stack: Vec<_> = graph.node_ids().map(|node_id| (node_id, graph)).collect();

    while let Some((node_id, graph)) = stack.pop() {
        let node = graph.get_node(node_id);

        if let Node::Theta(theta) = node {
            stack.extend(
                theta
                    .body()
                    .node_ids()
                    .map(|node_id| (node_id, theta.body())),
            );
        } else if let Node::Gamma(gamma) = node {
            stack.extend(
                gamma
                    .true_branch()
                    .node_ids()
                    .map(|node_id| (node_id, gamma.true_branch())),
            );
            stack.extend(
                gamma
                    .false_branch()
                    .node_ids()
                    .map(|node_id| (node_id, gamma.false_branch())),
            );
        }

        let input_desc = node.input_desc();
        let (input_effects, input_values) =
            graph
                .all_node_inputs(node_id)
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
            .all_output_ports()
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

    let elapsed = start_time.elapsed();
    tracing::debug!(
        target: "timings",
        "took {:#?} to validate graph",
        elapsed,
    );
}
