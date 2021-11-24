use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[clap(rename_all = "kebab-case")]
pub struct Args {
    /// The command to run
    #[clap(subcommand)]
    pub command: Command,

    /// The length of the program tape
    #[clap(long, default_value = "30000")]
    pub cells: u16,

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
    pub inline_constants: bool,
}

#[derive(Debug, Parser)]
#[clap(rename_all = "kebab-case")]
pub enum Command {
    /// Optimize and execute a brainfuck file
    Run {
        /// The file to run
        file: PathBuf,

        /// Disable optimizations
        #[clap(long)]
        no_opt: bool,
    },

    /// Optimize and run a brainfuck file along with all intermediate steps
    Debug {
        /// The file to debug
        file: PathBuf,

        /// Only run the final optimized program
        #[clap(long)]
        only_final_run: bool,

        /// Remove the interpreter's step limit
        #[clap(long)]
        no_step_limit: bool,
    },

    /// Step through a brainfuck program
    Debugger {
        /// The file to run
        file: PathBuf,
    },
}

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub struct Run {
    /// The file to execute
    pub file: PathBuf,

    /// The length of the program tape
    #[clap(long, default_value = "30000")]
    pub cells: u16,

    /// The maximum number of optimization iterations
    #[clap(long)]
    pub iteration_limit: Option<usize>,

    /// The interpreter's step limit
    #[clap(long)]
    pub step_limit: Option<usize>,
}

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub struct Debug {
    /// The file to debug
    pub file: PathBuf,

    /// The length of the program tape
    #[clap(long, default_value = "30000")]
    pub cells: u16,

    /// The maximum number of optimization iterations
    #[clap(long)]
    pub iteration_limit: Option<usize>,

    /// The interpreter's step limit
    #[clap(long, default_value = "300000")]
    pub step_limit: usize,

    /// Only run the final optimized program
    #[clap(long)]
    pub only_final_run: bool,

    /// Removes execution step limits
    #[clap(long)]
    pub no_step_limit: bool,

    /// Prints the final optimized IR
    #[clap(long)]
    pub print_output_ir: bool,
}
