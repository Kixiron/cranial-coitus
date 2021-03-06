use pretty::{Arena, DocAllocator, DocBuilder};
use std::{fmt::Write, time::Instant};
use tracing::{level_filters::STATIC_MAX_LEVEL, metadata::LevelFilter};

const RENDER_WIDTH: usize = 80;
pub(super) const INDENT_WIDTH: usize = 4;
pub(super) const COMMENT_ALIGNMENT_OFFSET: usize = 25;

// TODO: Colored output for the terminal
//       https://docs.rs/pretty/latest/pretty/trait.RenderAnnotated.html#impl-RenderAnnotated%3C%27_%2C%20ColorSpec%3E

#[derive(Debug, Clone, Copy)]
pub struct PrettyConfig {
    pub display_effects: bool,
    pub display_invocations: bool,
    pub total_instructions: Option<usize>,
    /// Log the time spent on pretty printing
    pub duration_logging: bool,
    /// If printed to an ansi-capable source, this will allow coloration
    pub colored: bool,
    pub show_lifetimes: bool,
    pub hide_const_assignments: bool,
}

impl PrettyConfig {
    pub fn minimal() -> Self {
        Self {
            display_effects: false,
            display_invocations: true,
            total_instructions: None,
            duration_logging: false,
            colored: true,
            show_lifetimes: false,
            hide_const_assignments: true,
        }
    }

    pub fn with_hide_const_assignments(mut self, hide_const_assignments: bool) -> Self {
        self.hide_const_assignments = hide_const_assignments;
        self
    }

    pub fn with_instrumented(mut self, total_instructions: usize) -> Self {
        self.total_instructions = Some(total_instructions);
        self
    }

    // pub const fn with_duration_logging(self, duration_logging: bool) -> Self {
    //     Self {
    //         duration_logging,
    //         ..self
    //     }
    // }

    // pub const fn with_color(self, colored: bool) -> Self {
    //     Self { colored, ..self }
    // }
}

pub trait Pretty {
    fn pretty_print(&self, config: PrettyConfig) -> String {
        let construction_start = Instant::now();

        let arena = Arena::<()>::new();
        let pretty = self.pretty(&arena, config);

        if STATIC_MAX_LEVEL >= LevelFilter::DEBUG && config.duration_logging {
            let elapsed = construction_start.elapsed();
            tracing::debug!(
                target: "timings",
                "took {:#?} to construct pretty printed ir",
                elapsed,
            );
        }

        let format_start = Instant::now();

        let mut output = String::with_capacity(4096);
        write!(output, "{}", pretty.1.pretty(RENDER_WIDTH))
            .expect("writing to a string should never fail");

        if STATIC_MAX_LEVEL >= LevelFilter::DEBUG && config.duration_logging {
            let elapsed = format_start.elapsed();
            tracing::debug!(
                target: "timings",
                "took {:#?} to construct format pretty printed ir",
                elapsed,
            );
        }

        output
    }

    fn pretty<'a, D, A>(&'a self, allocator: &'a D, config: PrettyConfig) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone;
}

pub mod pretty_utils {
    use crate::ir::{pretty_print::INDENT_WIDTH, Instruction, Pretty, PrettyConfig};
    use pretty::{DocAllocator, DocBuilder};

    pub fn binary<'a, D, A, L, R>(
        op: &'static str,
        lhs: &'a L,
        rhs: &'a R,
        allocator: &'a D,
        config: PrettyConfig,
    ) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
        L: Pretty,
        R: Pretty,
    {
        allocator
            .text(op)
            .append(allocator.space())
            .append(lhs.pretty(allocator, config))
            .append(allocator.text(","))
            .append(allocator.space())
            .append(rhs.pretty(allocator, config))
    }

    pub fn unary<'a, D, A, U>(
        op: &'static str,
        arg: &'a U,
        allocator: &'a D,
        config: PrettyConfig,
    ) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
        U: Pretty + 'static,
    {
        allocator
            .text(op)
            .append(allocator.space())
            .append(arg.pretty(allocator, config))
    }

    pub fn body_block<'a, D, A>(
        allocator: &'a D,
        config: PrettyConfig,
        indent: bool,
        block: &'a [Instruction],
    ) -> DocBuilder<'a, D, A>
    where
        D: DocAllocator<'a, A>,
        D::Doc: Clone,
        A: Clone,
    {
        let indent_width = if indent { INDENT_WIDTH } else { 0 };

        let instructions = block.iter().filter(|inst| {
            (config.show_lifetimes || !inst.is_lifetime_end())
                && !(config.hide_const_assignments
                    && inst
                        .as_assign()
                        .map(|assign| assign.value.is_const())
                        .unwrap_or(false))
        });
        let total_instructions = instructions.clone().count();

        if total_instructions == 0 {
            allocator.nil()
        } else {
            if indent {
                allocator.hardline()
            } else {
                allocator.nil()
            }
            .append(
                allocator
                    .intersperse(
                        instructions.map(|inst| inst.pretty(allocator, config)),
                        allocator.hardline(),
                    )
                    .indent(indent_width),
            )
            .append(allocator.hardline())
        }
    }
}
