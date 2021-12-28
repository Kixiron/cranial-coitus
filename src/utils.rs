use atty::Stream;
use similar::{Algorithm, TextDiff};
use std::{
    collections::BTreeSet,
    fmt::{self, Debug, Display},
    hash::{BuildHasherDefault, Hash},
    time::{Duration, Instant},
};
use tracing_subscriber::{
    fmt::TestWriter, prelude::__tracing_subscriber_SubscriberExt, util::SubscriberInitExt,
    EnvFilter,
};
use tracing_tree::HierarchicalLayer;
use xxhash_rust::xxh3::Xxh3;

#[cfg(test)]
use crate::{
    graph::{OutputPort, Rvsdg},
    lower_tokens, parse,
};

pub type HashSet<K> = std::collections::HashSet<K, BuildHasherDefault<Xxh3>>;
pub type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<Xxh3>>;

pub(crate) enum Element<T> {
    Single(T),
    Many(T, usize, usize),
}

impl<T> Debug for Element<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Element::Single(elem) => Debug::fmt(elem, f),
            Element::Many(elem, start, len) => {
                write!(f, "{:?} × {}..{}", elem, start, start + len)
            }
        }
    }
}

pub(crate) fn debug_collapse<T>(elements: &[T]) -> Vec<Element<T>>
where
    T: Copy + PartialEq,
{
    let (mut idx, mut output) = (0, Vec::with_capacity(elements.len()));

    while idx < elements.len() {
        let similar = elements[idx..]
            .iter()
            .take_while(|&elem| elem == &elements[idx]);

        let len = similar.count();

        if len >= 5 {
            output.push(Element::Many(elements[idx], idx, len));
            idx += len;
        } else {
            output.push(Element::Single(elements[idx]));
            idx += 1;
        }
    }

    output
}

pub fn percent_total(total: usize, subset: usize) -> f64 {
    let diff = (subset as f64 * 100.0) / total as f64;

    if diff.is_nan() || diff == -0.0 {
        0.0
    } else {
        diff
    }
}

pub fn diff_ir(old: &str, new: &str) -> String {
    let start_time = Instant::now();

    let diff = TextDiff::configure()
        .algorithm(Algorithm::Patience)
        .deadline(Instant::now() + Duration::from_secs(1))
        .diff_lines(old, new);

    let diff = format!("{}", diff.unified_diff());

    let elapsed = start_time.elapsed();
    tracing::debug!(
        target: "timings",
        "took {:#?} to diff ir",
        elapsed,
    );

    diff
}

#[non_exhaustive]
pub struct PerfEvent {
    start_time: Instant,
}

impl PerfEvent {
    pub fn new(event_name: &str) -> Self {
        superluminal_perf::begin_event(event_name);

        Self {
            start_time: Instant::now(),
        }
    }

    pub fn finish(self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Drop for PerfEvent {
    fn drop(&mut self) {
        superluminal_perf::end_event();
    }
}

#[cfg(test)]
pub fn compile_brainfuck_into(
    source: &str,
    graph: &mut Rvsdg,
    ptr: OutputPort,
    effect: OutputPort,
) -> (OutputPort, OutputPort) {
    let parsing_start = Instant::now();

    let span = tracing::info_span!("parsing");
    let tokens = span.in_scope(|| {
        tracing::info!("started parsing source code");
        let tokens = parse::parse(source);

        let elapsed = parsing_start.elapsed();
        tracing::info!("finished parsing in {:#?}", elapsed);

        tokens
    });

    let span = tracing::info_span!("rvsdg-building");
    span.in_scope(|| {
        tracing::info!("started building rvsdg");
        let graph_building_start = Instant::now();

        let (ptr, effect) = lower_tokens::lower_tokens(graph, ptr, effect, &tokens);

        let elapsed = graph_building_start.elapsed();
        tracing::info!("finished building rvsdg in {:#?}", elapsed);

        (ptr, effect)
    })
}

/// Uses a type's [`Display`] implementation for [`Debug`] printing it
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct DebugDisplay<T>(T);

impl<T> DebugDisplay<T> {
    /// Creates a new `DebugDisplay`
    pub fn new(inner: T) -> Self {
        Self(inner)
    }

    /// Gets the inner value from a `DebugDisplay`
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> Debug for DebugDisplay<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl<T> Display for DebugDisplay<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
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

pub(crate) trait AssertNone: Debug {
    fn debug_unwrap(self);

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

impl<T> AssertNone for Option<T>
where
    T: Debug,
{
    #[inline]
    #[track_caller]
    fn debug_unwrap(self) {
        if cfg!(debug_assertions) {
            self.unwrap();
        }
    }

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

pub trait OptionExt<T> {
    fn inspect<F>(self, inspect: F) -> Self
    where
        F: FnOnce(&T);
}

impl<T> OptionExt<T> for Option<T> {
    fn inspect<F>(self, inspect: F) -> Self
    where
        F: FnOnce(&T),
    {
        if let Some(value) = &self {
            inspect(value);
        }

        self
    }
}

pub(crate) fn set_logger() {
    let fmt_layer = HierarchicalLayer::new(2)
        // Don't use ansi codes if we're not printing to a console
        .with_ansi(atty::is(Stream::Stdout));

    let filter_layer = EnvFilter::try_from_env("COITUS_LOG")
        .or_else(|_| EnvFilter::try_new("off"))
        .unwrap();

    let registry = tracing_subscriber::registry();
    let _ = if cfg!(test) {
        // Use a logger that'll be captured by libtest if we're running
        // under a test harness
        registry
            .with(fmt_layer.with_writer(TestWriter::new()))
            .with(filter_layer)
            .try_init()
    } else {
        registry.with(fmt_layer).with(filter_layer).try_init()
    };
}

#[macro_export]
macro_rules! bvec {
    () => { ::std::collections::Vec::new() };

    ($elem:expr; $n:expr) => {{
        ::std::vec![::std::boxed::Box::new($elem); $n]
    }};

    ($($x:expr),+ $(,)?) => {{
        ::std::vec![$(::std::boxed::Box::new($x),)+]
    }};
}

#[macro_export]
macro_rules! vec_deque {
    () => { ::std::collections::VecDeque::new() };

    ($elem:expr; $n:expr) => {{
        let len: usize = $n;

        let mut vec = ::std::collections::VecDeque::with_capacity(len);
        for _ in 0..len {
            vec.push_back($x);
        }

        vec
    }};

    ($($elem:expr),+ $(,)?) => {{
        const LENGTH: usize = <[()]>::len(&[
            $( $crate::replace_expr!($elem; ()), )+
        ]);

        let mut vec = ::std::collections::VecDeque::with_capacity(LENGTH);
        $( vec.push_back($elem); )+
        vec
    }};
}

#[macro_export]
macro_rules! map {
    () => { $crate::utils::HashMap::new() };

    ($($key:expr => $value:expr),+ $(,)?) => {{
        const LENGTH: usize = <[()]>::len(&[
            $( $crate::replace_expr!($key; ()), )+
        ]);

        let mut map = $crate::utils::HashMap::with_capacity_and_hasher(
            LENGTH,
            ::std::hash::BuildHasherDefault::default(),
        );
        $( map.insert($key, $value); )+
        map
    }};
}

#[macro_export]
macro_rules! set {
    () => { $crate::utils::HashSet::new() };

    ($($key:expr),+ $(,)?) => {{
        const LENGTH: usize = <[()]>::len(&[
            $( $crate::replace_expr!($key; ()), )+
        ]);

        let mut set = $crate::utils::HashSet::with_capacity_and_hasher(
            LENGTH,
            ::std::hash::BuildHasherDefault::default(),
        );
        $( set.insert($key); )+
        set
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! replace_expr {
    ($expr:expr; $replacement:expr) => {
        $replacement
    };
}

// FIXME: Check for structural equivalence between the optimized graph
//        and an expected graph
#[macro_export]
macro_rules! test_opts {
    (
        $name:ident,
        $(default_passes = $default_passes:expr,)?
        $(passes = [$($pass:expr),* $(,)?],)?
        $(tape_size = $tape_size:expr,)?
        $(step_limit = $step_limit:expr,)?
        $(input = [$($input:expr),* $(,)?],)?
        output = $output:expr,
        $build:expr $(,)?
    ) => {
        #[test]
        fn $name() {
            #[allow(unused_imports)]
            use $crate::{
                graph::{Rvsdg, OutputPort, ThetaData, GammaData},
                interpreter::Machine,
                ir::{IrBuilder, Pretty, PrettyConfig},
                passes::{Pass, Dce},
                utils::{compile_brainfuck_into, HashSet},
            };
            use std::collections::VecDeque;

            $crate::utils::set_logger();

            let tape_size: usize = $crate::test_opts!(@tape_size $($tape_size)?);
            let step_limit: usize = $crate::test_opts!(@step_limit $($step_limit)?);

            let mut passes: Vec<Box<dyn Pass>> = vec![
                $($(Box::new($pass) as Box<dyn Pass>,)*)?
            ];
            $(
                if $default_passes {
                    passes = $crate::passes::default_passes(tape_size);
                }
            )?

            let build: fn(&mut Rvsdg, OutputPort) -> OutputPort = $build;

            let expected_output: Vec<u8> = $output.to_vec();

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

                let mut unoptimized_graph_ir = IrBuilder::new(false).translate(&graph);
                let unoptimized_text = unoptimized_graph_ir.pretty_print(PrettyConfig::minimal());

                let mut machine = Machine::new(step_limit, tape_size, input_func, output_func);
                machine
                    .execute(&mut unoptimized_graph_ir, false)
                    .expect("interpreter step limit reached");

                unoptimized_text
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
                    HashSet::with_hasher(Default::default()),
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

                let mut optimized_graph_ir = IrBuilder::new(false).translate(&graph);
                let optimized_text = optimized_graph_ir.pretty_print(PrettyConfig::minimal());

                let mut machine = Machine::new(step_limit, tape_size, input_func, output_func);
                machine
                    .execute(&mut optimized_graph_ir, false)
                    .expect("interpreter step limit reached");

                optimized_text
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

    (@tape_size $tape_size:expr) => { $tape_size };
    (@tape_size) => { 30_000 };

    (@step_limit $step_limit:expr) => { $step_limit };
    (@step_limit) => { 300_000 };
}
