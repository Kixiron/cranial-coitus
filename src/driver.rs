use crate::{
    graph::Rvsdg,
    interpreter::{ExecutionStats, Machine, StepLimitReached},
    ir::{Block, IrBuilder, Pretty},
    lower_tokens,
    parse::{self, Token},
    utils::PerfEvent,
};
use anyhow::{Context, Result};
use std::{
    collections::VecDeque,
    fs,
    io::{self, Read, Write},
    path::Path,
    time::Duration,
};

#[tracing::instrument(skip_all)]
pub fn build_graph(tokens: &[Token]) -> Rvsdg {
    tracing::info!(total_tokens = tokens.len(), "started building rvsdg");
    let event = PerfEvent::new("build-graph");

    let mut graph = Rvsdg::new();
    // Create the graph's start node
    let start = graph.start();

    // Get the starting effect and create the initial pointer (zero)
    let effect = start.effect();
    let ptr = graph.int(0).value();

    // Lower all of the program's tokens into the graph
    let (_ptr, effect) = lower_tokens::lower_tokens(&mut graph, ptr, effect, tokens);
    // Create the program's end node
    graph.end(effect);

    let elapsed = event.finish();
    tracing::info!("finished building rvsdg in {:#?}", elapsed);

    graph
}

#[tracing::instrument(skip(source))]
pub fn parse_source(file: &Path, source: &str) -> Box<[Token]> {
    let file = file.display();

    tracing::info!(source_len = source.len(), "started parsing '{}'", file);
    let event = PerfEvent::new("parsing");

    // Parse the program's source code into tokens
    let tokens = parse::parse(source);

    let elapsed = event.finish();
    tracing::info!(
        source_len = source.len(),
        total_tokens = tokens.len(),
        "finished parsing '{}' in {:#?}",
        file,
        elapsed,
    );

    tokens
}

#[tracing::instrument(skip(input, output, program))]
pub fn execute<I, O>(
    step_limit: usize,
    cells: usize,
    input: I,
    output: O,
    should_profile: bool,
    program: &mut Block,
) -> (Result<(), StepLimitReached>, ExecutionStats, Duration)
where
    I: FnMut() -> u8,
    O: FnMut(u8),
{
    tracing::info!(step_limit, cells, "started program execution");
    let event = PerfEvent::new("program-execution");

    // Execute the given program with the given input & output functions
    let mut machine = Machine::new(step_limit, cells, input, output);
    let result = machine.execute(program, should_profile).map(|_| ());

    let elapsed = event.finish();
    tracing::info!(
        step_limit,
        cells,
        "finished program execution in {:#?}",
        elapsed,
    );

    (result, machine.stats, elapsed)
}

#[tracing::instrument(skip(graph))]
pub fn sequentialize_graph(
    graph: &Rvsdg,
    dump_dir: Option<&Path>,
    total_instructions: Option<usize>,
) -> Result<(Block, String)> {
    // Sequentialize the graph into serial instructions
    let program = {
        tracing::info!("started sequentializing graph");
        let event = PerfEvent::new("sequentializing-graph");

        let sequential_code = IrBuilder::new().translate(graph);

        let elapsed = event.finish();
        tracing::info!("finished sequentializing graph in {:#?}", elapsed);

        sequential_code
    };

    // Pretty print the IR
    let program_ir = {
        tracing::info!("started pretty printing sequentialized graph");
        let event = PerfEvent::new("pretty-printing-sequentialized-graph");

        let pretty_printed = program.pretty_print(total_instructions);

        let elapsed = event.finish();
        tracing::info!(
            "finished pretty printing sequentialized graph in {:#?}",
            elapsed,
        );

        pretty_printed
    };

    // Dump the IR to file if requested
    if let Some(dump_dir) = dump_dir {
        tracing::debug!("dumping sequentialized graph to '{}'", dump_dir.display());

        fs::write(dump_dir, &program_ir).with_context(|| {
            format!(
                "failed to write sequentialized graph to '{}'",
                dump_dir.display(),
            )
        })?;
    }

    Ok((program, program_ir))
}

pub fn create_dump_dir(dump_dir: &Path) -> Result<()> {
    tracing::info!("creating dump directory at '{}'", dump_dir.display());

    // Delete anything that was previously within the folder (if there was any)
    let _ = fs::remove_dir_all(&dump_dir);

    // Create the dump directory and any required folders up to it
    fs::create_dir_all(&dump_dir)
        .with_context(|| format!("failed to create dump directory at {}", dump_dir.display()))?;

    Ok(())
}

pub fn array_input(input: &mut VecDeque<u8>) -> impl FnMut() -> u8 + '_ {
    move || {
        let (byte, was_empty) = input.pop_front().map_or((0, true), |byte| (byte, false));

        tracing::trace!(
            input_len = input.len(),
            was_empty,
            "popped input value {byte}, hex: {byte:#X}, binary: {byte:#08b}",
            byte = byte,
        );

        byte
    }
}

pub fn array_output(output: &mut Vec<u8>) -> impl FnMut(u8) + '_ {
    move |byte| {
        tracing::trace!(
            output_len = output.len() + 1,
            "pushed output value {byte}, hex: {byte:#X}, binary: {byte:#08b}",
            byte = byte,
        );
        output.push(byte);
    }
}

pub fn stdin_input() -> impl FnMut() -> u8 + 'static {
    move || {
        // FIXME: Lock once, move into closure
        let stdin_handle = io::stdin();
        let mut stdin = stdin_handle.lock();

        let mut buf = [0];
        stdin
            .read_exact(&mut buf)
            .expect("failed to read from stdin");
        let [byte] = buf;

        tracing::trace!(
            "read input value {byte}, hex: {byte:#X}, binary: {byte:#08b}",
            byte = byte,
        );

        byte
    }
}

pub fn stdout_output() -> impl FnMut(u8) + 'static {
    move |byte| {
        // FIXME: Lock once, move into closure
        let stdout_handle = io::stdout();
        let mut stdout = stdout_handle.lock();

        tracing::trace!(
            "wrote output value {byte}, hex: {byte:#X}, binary: {byte:#08b}",
            byte = byte,
        );

        stdout
            .write_all(&[byte])
            .expect("failed to write to stdout");

        // FIXME: Flush infrequently, somehow?
        stdout.flush().expect("failed to flush stdout");
    }
}
