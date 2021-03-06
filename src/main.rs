#![feature(
    stdsimd,
    option_zip,
    let_chains,
    new_uninit,
    try_blocks,
    cell_update,
    inline_const,
    entry_insert,
    array_chunks,
    drain_filter,
    array_windows,
    slice_ptr_get,
    slice_ptr_len,
    str_internals,
    portable_simd,
    adt_const_params,
    hash_drain_filter,
    vec_into_raw_parts,
    generic_const_exprs,
    local_key_cell_methods,
    maybe_uninit_write_slice,
    nonnull_slice_from_raw_parts
)]
#![allow(incomplete_features)]
#![deny(unsafe_op_in_unsafe_fn)]

#[macro_use]
mod utils;
mod args;
mod driver;
mod graph;
mod interpreter;
mod ir;
mod jit;
mod lower_tokens;
mod parse;
mod passes;
mod patterns;
mod tests;
mod values;

use crate::{
    args::{Args, Settings},
    graph::{EdgeKind, Node, NodeExt, Rvsdg},
    interpreter::EvaluationError,
    ir::{IrBuilder, Pretty, PrettyConfig},
    jit::Jit,
    parse::Parsed,
    utils::{HashMap, HashSet, PassName, PerfEvent},
    values::Ptr,
};
use anyhow::{Context, Result};
use clap::Parser;
use std::{
    cmp::max,
    collections::{hash_map::Entry, VecDeque},
    fmt::Write,
    fs::{self, File},
    io::{BufWriter, Write as _},
    panic,
    path::Path,
    time::{Duration, Instant},
};

// TODO: Generate ELK text files
// https://rtsys.informatik.uni-kiel.de/elklive/elkgraph.html
// https://github.com/eclipse/elk/pull/106
// https://rtsys.informatik.uni-kiel.de/elklive/examples.html
// https://rtsys.informatik.uni-kiel.de/elklive/elkgraph.html?compressedContent=IYGw5g9gTglgLgCwLYC4AEJgE8CmUcAmAUEQHYQE5qI5zBoDeRaLaOIA1gHQEz4DGcGBFLoAIgHkA6gDlmrAA7Q4AYREBnOFGAxScdegBiAJQCip+S3KUAMsABG7dVwWZ+OJDj3oARAEkZAGU-MVM0ADUAfQAVCQAFNAAJSJtTQ2ifSzQlKDgAQRAYMFJPPR4cADNgAFcQOHE-QOjjPwAhAFVo0zEs3XUYSkD2CpsICAVnYEEYADdgOBx0LWqcElYMB3Y0HwBxYCQkYEysnLg0XQVquABGRiz1gD1Trn7KdBkJY2jErIBfE+U51IlzgACY7utHs9Xos0B8vj91v91qc0Jp5jhIroIZCWE9lC8BrD4d8-mtWNYqMACAQcbjKXZHCBnK4ph4vPVtgFgqEIpEVKYZF1jEl+YLhZlcSxTgUiiUOeUqrVOWJGs02p1uuTcZgmds8jTjlLsoCQAh1HTjWh8blCW84Z9SVbWLqtj4zepJVLkVLUVBzZbjTa4HbiY7Ec6NnqfP7PfdWP949LAXMQCtA1Lg6H0IEJJ0I7ifSxEyjARAriDbkwg9CiTm806EwDcmi6AtIuWztXM7X7bn82SsoQwFQaHQuBcrrcALQAPjQ1IIXA9Q4II+oCFowAnwKu4LnC5pXFjq-Xi64qfTB7H287laIieHo8347vU7Q15f28nYKIT43W4vG2mLYp+gHou2naWn0RJDCAIxjBMXBYBASxQCsD5EEAA
fn main() -> Result<()> {
    utils::set_logger();

    let start_time = Instant::now();

    let args = Args::parse();
    tracing::info!(?args, "launched with cli arguments");

    match &args {
        Args::Run { file, settings } => run(settings, file, start_time),
        Args::Debug { file, settings } => debug(settings, file, start_time),
    }
}

// TODO: Clean this up
// TODO: User feedback
fn debug(settings: &Settings, file: &Path, start_time: Instant) -> Result<()> {
    let step_limit = settings.step_limit();
    let pretty_config =
        PrettyConfig::minimal().with_hide_const_assignments(!settings.dont_inline_constants);

    let source = fs::read_to_string(file).expect("failed to read file");

    let dump_dir = Path::new("./dumps").join(file.with_extension("").file_name().unwrap());
    driver::create_dump_dir(&dump_dir)?;

    // Parse the input program and turn it into a graph
    let tokens = driver::parse_source(file, &source);
    let mut graph = driver::build_graph(&tokens, settings.tape_len.get());

    // Sequentialize the input program
    let (mut input_program, input_program_ir) = driver::sequentialize_graph(
        settings,
        &graph,
        Some(&dump_dir.join("input.cir")),
        pretty_config,
    )?;
    let input_graph_stats = graph.stats();

    // Validate the initial program
    // Note: This happens *after* we print out the initial graph for better debugging
    validate(&graph);

    let unoptimized_execution = if !settings.no_run && settings.run_unoptimized_program {
        let compile_attempt: Result<Result<_>, _> = panic::catch_unwind(|| {
            let jit =
                Jit::new(settings, Some(&dump_dir), Some("input"))?.compile(&input_program)?;

            let mut tape = vec![0x00; settings.tape_len.get() as usize];
            let start = Instant::now();

            // Safety: Decidedly not safe in the slightest
            unsafe { jit.execute(&mut tape)? };

            let elapsed = start.elapsed();
            println!("Unoptimized jit finished execution in {:#?}", elapsed);

            Ok(())
        });

        match compile_attempt {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::error!("jit compilation attempt failed: {:?}", error);
            }
            Err(error) => {
                tracing::error!("jit compilation attempt panicked: {:?}", error);
            }
        }

        let result_path = dump_dir.join("input-result.txt");
        let mut result_file = File::create(&result_path)
            .with_context(|| format!("failed to create '{}'", result_path.display()))?;

        // FIXME: Allow user supplied input
        let mut input = vec_deque![b'1', b'0'];
        let input_vec: Vec<_> = input.iter().copied().collect();
        let input = driver::array_input(&mut input);

        let mut output_vec = Vec::new();
        let output = driver::array_output(&mut output_vec);

        let (result, tape, stats, execution_duration) = driver::execute(
            step_limit,
            settings.tape_len.get(),
            input,
            output,
            false,
            &mut input_program,
        );

        // FIXME: Utility function
        let input_str = String::from_utf8_lossy(&input_vec);
        writeln!(
            result_file,
            "----- Input -----\n{:?}\n-----\n{:?}\n-----\n{}",
            input_vec, input_str, input_str,
        )?;

        // FIXME: Utility function
        let output_str = String::from_utf8_lossy(&output_vec);
        writeln!(
            result_file,
            "----- Output -----\n{:?}\n-----\n{:?}\n-----\n{}",
            output_vec, output_str, output_str,
        )?;

        let tape_chars =
            utils::debug_collapse(&String::from_utf8_lossy(&tape).chars().collect::<Vec<_>>());
        let tape = utils::debug_collapse(&tape);

        writeln!(
            result_file,
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
    write!(evolution, ">>>>> input\n{}", input_program_ir).unwrap();

    let mut passes: Vec<_> = {
        let passes = passes::default_passes(&settings.pass_config());

        let mut pass_names = HashMap::with_capacity_and_hasher(passes.len(), Default::default());

        for (idx, pass) in passes.into_iter().enumerate() {
            for i in 0usize.. {
                let name = PassName::new(pass.pass_name(), i);

                // If no pass by this name we cna use it for the current pass
                if let Entry::Vacant(entry) = pass_names.entry(name) {
                    entry.insert((idx, pass));
                    break;
                }
            }
        }

        let mut passes: Vec<_> = pass_names.into_iter().collect();
        passes.sort_unstable_by_key(|&(_, (idx, _))| idx);
        passes
            .into_iter()
            .map(|(name, (_, pass))| (name, pass))
            .collect()
    };

    let (mut pass_stats, mut pass_num, mut stack, mut visited, mut buffer, mut previous_program_ir) = (
        HashMap::with_capacity_and_hasher(passes.len(), Default::default()),
        1,
        VecDeque::new(),
        HashSet::default(),
        Vec::new(),
        input_program_ir.clone(),
    );

    if !settings.disable_optimizations {
        let mut pass_name_buf = String::with_capacity(128);

        let mut dump_files = {
            let mut dump_files =
                HashMap::with_capacity_and_hasher(passes.len(), Default::default());

            let (mut file_name_buf, mut pass_path_buf) =
                (String::with_capacity(128), dump_dir.clone());
            for (pass_idx, &(name, _)) in passes.iter().enumerate() {
                pass_path_buf.push(name.name());
                if pass_num == 1 {
                    fs::create_dir_all(&pass_path_buf)?;
                }

                write!(file_name_buf, "{}.{}.input.cir", pass_num, pass_idx).unwrap();
                pass_path_buf.push(&file_name_buf);
                let input_path = File::create(&pass_path_buf).with_context(|| {
                    format!(
                        "failed to create dump file '{}' for pass '{}'",
                        pass_path_buf.display(),
                        name,
                    )
                })?;
                file_name_buf.clear();
                pass_path_buf.pop();

                write!(file_name_buf, "{}.{}.cir", pass_num, pass_idx).unwrap();
                pass_path_buf.push(&file_name_buf);
                let output_path = File::create(&pass_path_buf).with_context(|| {
                    format!(
                        "failed to create dump file '{}' for pass '{}'",
                        pass_path_buf.display(),
                        name,
                    )
                })?;
                file_name_buf.clear();
                pass_path_buf.pop();

                write!(file_name_buf, "{}.{}.diff", pass_num, pass_idx).unwrap();
                pass_path_buf.push(&file_name_buf);
                let diff_path = File::create(&pass_path_buf).with_context(|| {
                    format!(
                        "failed to create dump file '{}' for pass '{}'",
                        pass_path_buf.display(),
                        name,
                    )
                })?;
                file_name_buf.clear();
                pass_path_buf.pop();

                dump_files.insert(
                    name,
                    (
                        BufWriter::new(input_path),
                        BufWriter::new(output_path),
                        BufWriter::new(diff_path),
                    ),
                );
                pass_path_buf.pop();
            }

            dump_files
        };

        loop {
            let mut changed = false;

            for (pass_idx, &mut (pass_name, ref mut pass)) in passes.iter_mut().enumerate() {
                let span = tracing::info_span!("pass", pass = %pass_name);
                span.in_scope(|| {
                    tracing::info!("running {} (pass #{}.{})", pass_name, pass_num, pass_idx,);

                    write!(pass_name_buf, "{}", pass_name).unwrap();
                    let event = PerfEvent::new(&pass_name_buf);

                    let pass_made_changes =
                        pass.visit_graph_inner(&mut graph, &mut stack, &mut visited, &mut buffer);
                    let elapsed = event.finish();

                    changed |= pass_made_changes;

                    // Update the pass stats
                    let (activations, duration) =
                        pass_stats.entry(pass_name).or_insert((0, Duration::ZERO));
                    *activations += pass_made_changes as usize;
                    *duration += elapsed;

                    tracing::info!(
                        "finished running {} in {:#?} (pass #{}.{}, {})",
                        pass_name,
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
                        driver::sequentialize_graph(settings, &graph, None, pretty_config)?;

                    let diff = utils::diff_ir(&previous_program_ir, &output_program_ir);

                    if !diff.is_empty() {
                        let (input_file, output_file, diff_file) =
                            dump_files.get_mut(&pass_name).unwrap();

                        input_file.write_all(previous_program_ir.as_bytes())?;
                        output_file.write_all(output_program_ir.as_bytes())?;
                        diff_file.write_all(diff.as_bytes())?;

                        write!(
                            evolution,
                            ">>>>> {}-{}.{}\n{}",
                            pass_name, pass_num, pass_idx, diff,
                        )?;
                    }

                    previous_program_ir = output_program_ir;
                    pass_name_buf.clear();

                    Result::<()>::Ok(())
                })?;
            }

            pass_num += 1;
            if !changed || pass_num >= settings.iteration_limit.unwrap_or(usize::MAX) {
                break;
            }
        }
    }

    let elapsed = start_time.elapsed();
    {
        let mut reports = Vec::with_capacity(passes.len());
        let mut max_len = 0;

        for (pass_name, pass) in &passes {
            let mut report: Vec<_> = pass.report().into_iter().collect();
            report.sort_unstable_by_key(|&(aspect, _)| aspect);

            if let Some(longest) = report.iter().map(|(aspect, _)| aspect.len()).max() {
                max_len = max(max_len, longest);
            }

            reports.push((pass_name, report));
        }

        let mut report_file = BufWriter::new(
            File::create(dump_dir.join("pass-report.txt"))
                .context("failed to create report file")?,
        );
        for (pass_name, report) in &reports {
            writeln!(report_file, "{}", pass_name)?;

            let (activations, duration) = pass_stats.get(pass_name).copied().unwrap_or_default();
            writeln!(
                report_file,
                "{:<padding$} : {}",
                "activations",
                activations,
                padding = max_len,
            )?;
            writeln!(
                report_file,
                "{:<padding$} : {:#?}",
                "duration",
                duration,
                padding = max_len,
            )?;

            if !report.is_empty() {
                for (aspect, total) in report {
                    writeln!(
                        report_file,
                        "{:<padding$} : {}",
                        aspect,
                        total,
                        padding = max_len,
                    )?;
                }
            }

            writeln!(report_file)?;
        }

        let no_info = reports.iter().filter(|(_, report)| report.is_empty());
        if no_info.clone().count() != 0 {
            writeln!(report_file, "No info for the following passes:")?;

            for (pass, _) in no_info {
                writeln!(report_file, "  - {}", pass)?;
            }
        }

        report_file.flush().context("failed to flush report file")?;
    }

    let output_graph_stats = graph.stats();
    let stats_difference = input_graph_stats.difference(output_graph_stats);
    let (mut output_program, output_program_ir) = driver::sequentialize_graph(
        settings,
        &graph,
        Some(&dump_dir.join("output.cir")),
        pretty_config.with_instrumented(output_graph_stats.instructions),
    )?;

    if settings.print_output_ir {
        print!("{}", output_program_ir);
    }

    let io_diff = utils::diff_ir(&input_program_ir, &output_program_ir);
    if !io_diff.is_empty() {
        fs::write(dump_dir.join("input-output.diff"), io_diff).unwrap();
    }

    let result_path = dump_dir.join("output-result.txt");
    let mut result_file = File::create(&result_path)
        .with_context(|| format!("failed to create '{}'", result_path.display()))?;

    // FIXME: Utility function
    let mut change = format!(
        "Optimized Program (took {} iterations and {:#?})\n\
         Input:\n  \
           instructions : {}\n  \
           branches     : {}\n  \
           loops        : {}\n  \
           loads        : {}\n  \
           stores       : {}\n  \
           constants    : {}\n  \
           io ops       : {}\n  \
           scans        : {}\n\
         Output:\n  \
           instructions : {}\n  \
           branches     : {}\n  \
           loops        : {}\n  \
           loads        : {}\n  \
           stores       : {}\n  \
           constants    : {}\n  \
           io ops       : {}\n  \
           scans        : {}\n\
         Change:\n  \
           instructions : {:>+6.02}%\n  \
           branches     : {:>+6.02}%\n  \
           loops        : {:>+6.02}%\n  \
           loads        : {:>+6.02}%\n  \
           stores       : {:>+6.02}%\n  \
           constants    : {:>+6.02}%\n  \
           io ops       : {:>+6.02}%\n  \
           scans        : {:>+6.02}%\n\n",
        pass_num,
        elapsed,
        input_graph_stats.instructions,
        input_graph_stats.branches,
        input_graph_stats.loops,
        input_graph_stats.loads,
        input_graph_stats.stores,
        input_graph_stats.constants,
        input_graph_stats.io_ops,
        input_graph_stats.scans,
        output_graph_stats.instructions,
        output_graph_stats.branches,
        output_graph_stats.loops,
        output_graph_stats.loads,
        output_graph_stats.stores,
        output_graph_stats.constants,
        output_graph_stats.io_ops,
        output_graph_stats.scans,
        stats_difference.instructions,
        stats_difference.branches,
        stats_difference.loops,
        stats_difference.loads,
        stats_difference.stores,
        stats_difference.constants,
        stats_difference.io_ops,
        stats_difference.scans,
    );

    if !settings.no_run {
        if settings.interpreter {
            let (optimized_stats, optimized_execution_duration) = {
                // FIXME: Allow user supplied input
                let mut input = vec_deque![b'1', b'0'];
                let input_vec: Vec<_> = input.iter().copied().collect();
                let input = driver::array_input(&mut input);

                let mut output_vec = Vec::new();
                let output = driver::array_output(&mut output_vec);

                let (result, tape, stats, execution_duration) = driver::execute(
                    step_limit,
                    settings.tape_len.get(),
                    input,
                    output,
                    true,
                    &mut output_program,
                );

                let input_str = String::from_utf8_lossy(&input_vec);
                writeln!(
                    result_file,
                    "----- Input -----\n{:?}\n-----\n{:?}\n-----\n{}",
                    input_vec, input_str, input_str,
                )?;

                // FIXME: Utility function
                let output_str = String::from_utf8_lossy(&output_vec);
                writeln!(
                    result_file,
                    "----- Output -----\n{:?}\n-----\n{:?}\n-----\n{}",
                    output_vec, output_str, output_str,
                )?;

                let tape_chars = utils::debug_collapse(
                    &String::from_utf8_lossy(&tape).chars().collect::<Vec<_>>(),
                );
                let tape = utils::debug_collapse(&tape);

                writeln!(
                    result_file,
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

            if let Some((unoptimized_stats, unoptimized_execution_duration)) = unoptimized_execution
            {
                writeln!(
                    change,
                    "Finished unoptimized execution in {:#?}\n\
                     Unoptimized execution stats:\n{}",
                    unoptimized_execution_duration,
                    unoptimized_stats.display(),
                )?;
            }

            writeln!(
                change,
                "Finished optimized execution in {:#?}\n\
                 Optimized execution stats:\n{}",
                optimized_execution_duration,
                optimized_stats.display(),
            )?;

            print!("{}", change);
            fs::write(dump_dir.join("change.txt"), change)?;

            let annotated_program = output_program
                .pretty_print(pretty_config.with_instrumented(optimized_stats.instructions));
            fs::write(dump_dir.join("annotated_output.cir"), annotated_program)?;
        } else {
            print!("{}", change);
            fs::write(dump_dir.join("change.txt"), change)?;
        }

        println!("Executing...");

        let jit = Jit::new(settings, Some(&dump_dir), Some("output"))?.compile(&output_program)?;

        let mut tape = vec![0x00; settings.tape_len.get() as usize];
        let start = Instant::now();

        // Safety: It probably isn't lol, my codegen is garbage
        unsafe { jit.execute(&mut tape)? };

        let elapsed = start.elapsed();
        println!("\nOptimized jit finished execution in {:#?}", elapsed);
        println!("{:?}", utils::debug_collapse(&tape));
    } else {
        print!("{}", change);
        fs::write(dump_dir.join("change.txt"), change)?;
    }

    Ok(())
}

fn run(settings: &Settings, file: &Path, start_time: Instant) -> Result<()> {
    let step_limit = settings.step_limit();
    let contents = fs::read_to_string(file).expect("failed to read file");

    let mut graph = {
        let span = tracing::info_span!("parsing");
        let tokens = span.in_scope(|| {
            tracing::info!("started parsing {}", file.display());

            let Parsed {
                tokens,
                source_len,
                total_tokens,
                deepest_nesting,
            } = parse::parse(&contents);

            let elapsed = start_time.elapsed();
            tracing::info!(
                source_len,
                total_tokens,
                deepest_nesting,
                "finished parsing {} in {:#?}",
                file.display(),
                elapsed,
            );

            tokens
        });

        tracing::info!("started building rvsdg");
        let graph_building_start = Instant::now();

        let mut graph = Rvsdg::new();
        let start = graph.start();

        let effect = start.effect();
        let ptr = graph.int(Ptr::zero(settings.tape_len.get())).value();

        let (_ptr, effect) = lower_tokens::lower_tokens(&mut graph, ptr, effect, &tokens);
        graph.end(effect);

        let elapsed = graph_building_start.elapsed();
        tracing::info!("finished building rvsdg in {:#?}", elapsed);

        graph
    };

    let input_stats = graph.stats();
    validate(&graph);

    let opt_iters = if settings.disable_optimizations {
        0
    } else {
        driver::run_opt_passes(
            &mut graph,
            settings.iteration_limit.unwrap_or(usize::MAX),
            &settings.pass_config(),
            Some(&mut HashMap::default()),
        )
    };

    let mut program = IrBuilder::new(!settings.dont_inline_constants).translate(&graph);
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
           io ops       : {}\n  \
           scans        : {}\n\
         Output:\n  \
           instructions : {}\n  \
           branches     : {}\n  \
           loops        : {}\n  \
           loads        : {}\n  \
           stores       : {}\n  \
           constants    : {}\n  \
           io ops       : {}\n  \
           scans        : {}\n\
         Change:\n  \
           instructions : {:>+6.02}%\n  \
           branches     : {:>+6.02}%\n  \
           loops        : {:>+6.02}%\n  \
           loads        : {:>+6.02}%\n  \
           stores       : {:>+6.02}%\n  \
           constants    : {:>+6.02}%\n  \
           io ops       : {:>+6.02}%\n  \
           scans        : {:>+6.02}%\n\
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
        input_stats.scans,
        output_stats.instructions,
        output_stats.branches,
        output_stats.loops,
        output_stats.loads,
        output_stats.stores,
        output_stats.constants,
        output_stats.io_ops,
        output_stats.scans,
        difference.instructions,
        difference.branches,
        difference.loops,
        difference.loads,
        difference.stores,
        difference.constants,
        difference.io_ops,
        difference.scans,
    );

    let input = driver::stdin_input();
    let output = driver::stdout_output();
    let (result, _, stats, execution_duration) = driver::execute(
        step_limit,
        settings.tape_len.get(),
        input,
        output,
        true,
        &mut program,
    );

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
