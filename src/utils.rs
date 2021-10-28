use std::{
    collections::{BTreeSet, HashSet},
    fmt::Debug,
    hash::Hash,
};

pub(crate) trait AssertNone: Debug {
    fn unwrap_none(&self);

    #[inline]
    #[track_caller]
    fn debug_unwrap_none(&self) {
        if cfg!(debug_assertions) {
            self.unwrap_none();
        }
    }

    fn expect_none(&self, message: &str);

    #[inline]
    #[track_caller]
    fn debug_expect_none(&self, message: &str) {
        if cfg!(debug_assertions) {
            self.expect_none(message);
        }
    }
}

pub trait Set<K> {
    fn contains(&self, value: &K) -> bool;

    fn is_empty(&self) -> bool;
}

impl<K> Set<K> for HashSet<K>
where
    K: Eq + Hash,
{
    #[inline]
    fn contains(&self, value: &K) -> bool {
        HashSet::contains(self, value)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        HashSet::is_empty(self)
    }
}

impl<K> Set<K> for BTreeSet<K>
where
    K: Ord,
{
    #[inline]
    fn contains(&self, value: &K) -> bool {
        BTreeSet::contains(self, value)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        BTreeSet::is_empty(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct SingletonSet<K>(K);

impl<K> SingletonSet<K> {
    pub const fn new(element: K) -> Self {
        Self(element)
    }

    pub fn into_inner(self) -> K {
        self.0
    }
}

impl<K> Set<K> for SingletonSet<K>
where
    K: PartialEq,
{
    #[inline]
    fn contains(&self, value: &K) -> bool {
        value == &self.0
    }

    #[inline]
    fn is_empty(&self) -> bool {
        false
    }
}

impl<T> AssertNone for Option<T>
where
    T: Debug,
{
    #[inline]
    #[track_caller]
    fn unwrap_none(&self) {
        if self.is_some() {
            panic_none(self)
        }
    }

    #[inline]
    #[track_caller]
    fn expect_none(&self, message: &str) {
        if self.is_some() {
            panic_none_with_message(self, message)
        }
    }
}

#[cold]
#[track_caller]
#[inline(never)]
fn panic_none(value: &dyn Debug) -> ! {
    panic!("unwrapped {:?} when `None` was expected", value)
}

#[cold]
#[track_caller]
#[inline(never)]
fn panic_none_with_message(value: &dyn Debug, message: &str) -> ! {
    panic!(
        "unwrapped {:?} when `None` was expected: {}",
        value, message
    )
}

// FIXME: Check for structural equivalence between the optimized graph
//        and an expected graph
#[macro_export]
macro_rules! test_opts {
    (
        $name:ident,
        passes = [$($pass:expr),+ $(,)?],
        $(tape_size = $tape_size:literal,)?
        $(step_limit = $step_limit:literal,)?
        $(input = [$($input:expr),* $(,)?],)?
        output = [$($output:expr),* $(,)?],
        $build:expr $(,)?
    ) => {
        #[test]
        fn $name() {
            use crate::{
                graph::{Rvsdg, OutputPort},
                interpreter::Machine,
                ir::{IrBuilder, Pretty},
                passes::Pass,
            };
            use std::collections::{BTreeSet, VecDeque};

            crate::set_logger();

            let tape_size: usize = $crate::test_opts!(@tape_size $($tape_size)?);
            let step_limit: usize = $crate::test_opts!(@step_limit $($step_limit)?);
            let mut passes: Vec<Box<dyn Pass>> = vec![
                $(Box::new($pass) as Box<dyn Pass>,)+
            ];
            let build: fn(&mut Rvsdg, OutputPort) -> OutputPort = $build;

            let expected_output: Vec<u8> = vec![$($output,)*];

            let mut graph = Rvsdg::new();
            let start = graph.start();
            let effect = build(&mut graph, start.effect());
            graph.end(effect);

            let mut unoptimized_output: Vec<u8> = Vec::new();
            let unoptimized_ir = {
                let mut input_bytes: Vec<u8> = vec![$($($input,)*)?];
                let input_func = move || if input_bytes.is_empty() {
                    0
                } else {
                    input_bytes.remove(0)
                };
                let output_func = |byte| unoptimized_output.push(byte);

                let unoptimized_graph_ir = IrBuilder::new().translate(&graph);
                let mut machine = Machine::new(step_limit, tape_size, input_func, output_func);
                machine
                    .execute(&unoptimized_graph_ir)
                    .expect("interpreter step limit reached");

                unoptimized_graph_ir.pretty_print()
            };

            let output_str = String::from_utf8_lossy(&unoptimized_output);
            tracing::info!(
                "produced unoptimized input:\n\
                 output (bytes)         : {:?}\n\
                 output (utf8, escaped) : {:?}\n\
                 output (utf8)          : {}\n\
                 ir dump:\n{}",
                unoptimized_output,
                output_str,
                output_str,
                unoptimized_ir,
            );

            assert_eq!(unoptimized_output, expected_output);

            let mut optimized_output: Vec<u8> = Vec::new();
            let optimized_ir = {
                let (mut stack, mut visited, mut buffer) = (
                    VecDeque::new(),
                    BTreeSet::new(),
                    Vec::new(),
                );

                loop {
                    let mut changed = false;

                    for pass in &mut passes {
                        changed |= pass.visit_graph_inner(
                            &mut graph,
                            &mut stack,
                            &mut visited,
                            &mut buffer,
                        );
                        pass.reset();
                        stack.clear();
                    }

                    if !changed {
                        break;
                    }
                }

                let mut input_bytes: Vec<u8> = vec![$($($input,)*)?];
                let input_func = move || if input_bytes.is_empty() {
                    0
                } else {
                    input_bytes.remove(0)
                };
                let output_func = |byte| optimized_output.push(byte);

                let optimized_graph_ir = IrBuilder::new().translate(&graph);
                let mut machine = Machine::new(step_limit, tape_size, input_func, output_func);
                machine
                    .execute(&optimized_graph_ir)
                    .expect("interpreter step limit reached");

                optimized_graph_ir.pretty_print()
            };

            let output_str = String::from_utf8_lossy(&optimized_output);
            tracing::info!(
                "produced optimized input:\n\
                 output (bytes)         : {:?}\n\
                 output (utf8, escaped) : {:?}\n\
                 output (utf8)          : {}\n\
                 ir dump:\n{}",
                optimized_output,
                output_str,
                output_str,
                optimized_ir,
            );

            assert_eq!(
                optimized_output,
                unoptimized_output,
                "graphs produced different output\n\
                 unoptimized:\n{}\n\n\
                 optimized:\n{}\n",
                unoptimized_ir,
                optimized_ir,
            );
        }
    };

    (@tape_size $tape_size:literal) => { $tape_size };
    (@tape_size) => { 30_000 };

    (@step_limit $step_limit:literal) => { $step_limit };
    (@step_limit) => { 300_000 };
}
