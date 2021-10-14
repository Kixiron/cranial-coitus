use clap::Clap;
use std::path::PathBuf;

#[derive(Clap)]
pub struct Args {
    pub file: PathBuf,

    /// The length of tape to optimize based off of
    #[clap(long, default_value = "30000")]
    pub cells: u16,

    /// The maximum number of optimization iterations to run
    #[clap(long)]
    pub iteration_limit: Option<usize>,
}
