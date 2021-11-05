use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum Args {
    Run(Run),
    Debug(Debug),
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
