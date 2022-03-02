use crate::{
    args::Args,
    graph::Rvsdg,
    interpreter::{EvaluationError, ExecutionStats, Machine},
    ir::{Block, IrBuilder, Pretty, PrettyConfig},
    lower_tokens,
    parse::{self, Token},
    passes,
    utils::{HashSet, PerfEvent},
    values::{Cell, Ptr},
};
use anyhow::{Context, Result};
use std::{
    collections::VecDeque,
    fs,
    io::{self, Read, Write},
    path::Path,
    time::{Duration, Instant},
};

#[tracing::instrument(skip(args))]
pub fn debugger(args: &Args, file: &Path, start_time: Instant) -> Result<()> {
    let cells = args.tape_len as usize;
    let step_limit = args.step_limit.unwrap_or(usize::MAX);

    let source = fs::read_to_string(file).expect("failed to read file");
    let tokens = parse_source(file, &source);

    let mut graph = build_graph(tokens, args.tape_len);
    run_opt_passes(
        &mut graph,
        args.tape_len,
        args.iteration_limit.unwrap_or(usize::MAX),
    );

    let ir = IrBuilder::new(args.inline_constants)
        .translate(&graph)
        .pretty_print(PrettyConfig::minimal());

    debugger_tui(cells, &ir)?;

    Ok(())
}

#[tracing::instrument(skip_all)]
pub fn run_opt_passes(graph: &mut Rvsdg, cells: u16, iteration_limit: usize) -> usize {
    let mut passes = passes::default_passes(cells);
    let (mut pass_num, mut stack, mut visited, mut buffer) = (
        1,
        VecDeque::new(),
        HashSet::with_hasher(Default::default()),
        Vec::new(),
    );

    loop {
        let mut changed = false;

        for (pass_idx, pass) in passes.iter_mut().enumerate() {
            let span = tracing::info_span!("optimization-pass", pass = pass.pass_name());
            span.in_scope(|| {
                tracing::info!(
                    "running {} (pass #{}.{})",
                    pass.pass_name(),
                    pass_num,
                    pass_idx,
                );

                let event = PerfEvent::new(pass.pass_name());
                changed |= pass.visit_graph_inner(graph, &mut stack, &mut visited, &mut buffer);
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
            });
        }

        pass_num += 1;
        if !changed || pass_num >= iteration_limit {
            break;
        }
    }

    for pass in &passes {
        pass.report();
    }

    pass_num
}

pub fn debugger_tui(cell_len: usize, program: &str) -> Result<()> {
    use tui::{
        backend::CrosstermBackend,
        layout::{Alignment, Constraint, Direction, Layout},
        text::Text,
        widgets::{Block, Borders, Cell, Paragraph, Row, Table, Wrap},
        Terminal,
    };

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend).context("failed to create terminal instance")?;
    terminal.clear().context("failed to clear terminal")?;

    terminal
        .draw(|frame| {
            let frame_size = frame.size();

            let [tape_size, program_size]: [_; 2] = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([Constraint::Percentage(30), Constraint::Percentage(70)].as_ref())
                .split(frame_size)
                .try_into()
                .expect("layout split into three chunks");

            let tape = Table::new(vec![Row::new(
                (0..20).map(|_| Cell::from("0x00")).collect::<Vec<_>>(),
            )])
            .widths(&[Constraint::Percentage(100)])
            .column_spacing(0);
            frame.render_widget(tape, tape_size);

            let program = Paragraph::new(Text::raw(program))
                .alignment(Alignment::Left)
                .wrap(Wrap { trim: false })
                .block(
                    Block::default()
                        .title_alignment(Alignment::Left)
                        .borders(Borders::ALL),
                );
            frame.render_widget(program, program_size);
        })
        .context("failed to draw frame")?;

    Ok(())
}

#[tracing::instrument(skip_all)]
pub fn build_graph<T>(tokens: T, tape_len: u16) -> Rvsdg
where
    T: AsRef<[Token]>,
{
    let tokens = tokens.as_ref();

    tracing::info!(total_tokens = tokens.len(), "started building rvsdg");
    let event = PerfEvent::new("build-graph");

    let mut graph = Rvsdg::new();
    // Create the graph's start node
    let start = graph.start();

    // Get the starting effect and create the initial pointer (zero)
    let effect = start.effect();
    let ptr = graph.int(Ptr::zero(tape_len)).value();

    // Lower all of the program's tokens into the graph
    let (_ptr, effect) = lower_tokens::lower_tokens(&mut graph, ptr, effect, tokens, tape_len);
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
    tape_len: u16,
    input: I,
    output: O,
    should_profile: bool,
    program: &mut Block,
) -> (
    Result<(), EvaluationError>,
    Vec<u8>,
    ExecutionStats,
    Duration,
)
where
    I: FnMut() -> u8,
    O: FnMut(u8),
{
    tracing::info!(step_limit, tape_len, "started program execution");
    let event = PerfEvent::new("program-execution");

    // Execute the given program with the given input & output functions
    let mut machine = Machine::new(step_limit, tape_len, input, output);
    let result = machine.execute(program, should_profile).map(|_| ());

    let elapsed = event.finish();
    tracing::info!(
        step_limit,
        tape_len,
        "finished program execution in {:#?}",
        elapsed,
    );

    let tape = machine
        .tape
        .iter()
        .map(|value| value.map_or(0, Cell::into_inner))
        .collect();

    (result, tape, machine.stats, elapsed)
}

#[tracing::instrument(skip(graph))]
pub fn sequentialize_graph(
    args: &Args,
    graph: &Rvsdg,
    dump_dir: Option<&Path>,
    config: PrettyConfig,
) -> Result<(Block, String)> {
    // Sequentialize the graph into serial instructions
    let program = {
        tracing::debug!("started sequentializing graph");
        let event = PerfEvent::new("sequentializing-graph");

        let sequential_code = IrBuilder::new(args.inline_constants).translate(graph);

        let elapsed = event.finish();
        tracing::debug!("finished sequentializing graph in {:#?}", elapsed);

        sequential_code
    };

    // Pretty print the IR
    let program_ir = {
        tracing::debug!("started pretty printing sequentialized graph");
        let event = PerfEvent::new("pretty-printing-sequentialized-graph");

        let pretty_printed = program.pretty_print(config);

        let elapsed = event.finish();
        tracing::debug!(
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
