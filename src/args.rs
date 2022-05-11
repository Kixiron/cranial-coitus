use clap::Parser;
use std::{num::NonZeroU16, path::PathBuf};

use crate::passes::PassConfig;

#[derive(Debug, Parser)]
#[clap(rename_all = "kebab-case")]
pub enum Args {
    /// Optimize and execute a brainfuck file
    Run {
        /// The file to run
        file: PathBuf,

        #[clap(flatten)]
        settings: Settings,
    },

    /// Optimize and run a brainfuck file along with all intermediate steps
    Debug {
        /// The file to debug
        file: PathBuf,

        #[clap(flatten)]
        settings: Settings,
    },
}

#[derive(Debug, Parser)]
#[clap(rename_all = "kebab-case")]
pub struct Settings {
    /// The length of the program tape
    #[clap(long, default_value = "30000")]
    pub tape_len: NonZeroU16,

    /// The maximum number of optimization iterations
    #[clap(long)]
    pub iteration_limit: Option<usize>,

    /// The interpreter's step limit
    #[clap(long)]
    pub step_limit: Option<usize>,

    /// Prints the final optimized IR
    #[clap(long)]
    pub print_output_ir: bool,

    /// Whether to inline constant values or give them their own
    /// assignments in the output IR
    #[clap(long)]
    pub dont_inline_constants: bool,

    /// If this is set, then tape wrapping is disabled. This allows
    /// greater optimizations and forbids programs that overflow or underflow
    /// the program tape pointer, meaning that instead of wrapping around to
    /// opposite side of the tape when the tape pointer is above the tape's length
    /// or below zero the operation will instead be UB
    #[clap(long)]
    pub tape_wrapping_ub: bool,

    /// If this is set, then cell wrapping is disabled. This allows
    /// greater optimizations and forbids programs that overflow
    /// or underflow within cells
    #[clap(long)]
    pub cell_wrapping_ub: bool,

    /// Disables all optimizations
    #[clap(long)]
    pub disable_optimizations: bool,

    /// Run the unoptimized program before optimizing and running it again
    #[clap(long)]
    pub run_unoptimized_program: bool,

    /// Remove the interpreter's step limit
    #[clap(long)]
    pub no_step_limit: bool,

    #[clap(long)]
    pub no_run: bool,

    #[clap(long)]
    pub interpreter: bool,
}

impl Settings {
    pub fn step_limit(&self) -> usize {
        if self.no_step_limit {
            usize::MAX
        } else {
            self.step_limit.unwrap_or(usize::MAX)
        }
    }

    pub fn pass_config(&self) -> PassConfig {
        PassConfig::new(
            self.tape_len.get(),
            !self.tape_wrapping_ub,
            !self.cell_wrapping_ub,
        )
    }
}
