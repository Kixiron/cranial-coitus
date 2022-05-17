#![cfg(test)]

use crate::{
    args::Settings,
    driver,
    graph::Rvsdg,
    ir::PrettyConfig,
    jit::Jit,
    utils::{compile_brainfuck_into, ByteVec},
    values::Ptr,
};
use std::num::NonZeroU16;

test_opts! {
    beer,
    use_default_passes = true,
    output = include_bytes!("corpus/beer.out"),
    |graph, effect, tape_len| {
        let ptr = graph.int(Ptr::zero(tape_len)).value();
        let (_, effect) = compile_brainfuck_into(
            include_str!("../../examples/beer.bf"),
            graph,
            ptr,
            effect,
        );

        effect
    },
}

#[test]
fn rot13() {
    crate::utils::set_logger();

    let tape_len = 30000;
    let input = b"~mlk zyx";
    let expected_output = b"~zyx mlk";

    let settings = Settings {
        tape_len: NonZeroU16::new(tape_len).unwrap(),
        tape_wrapping_ub: false,
        cell_wrapping_ub: false,
        ..Default::default()
    };

    let mut graph = Rvsdg::new();
    {
        let start = graph.start();
        let ptr = graph.int(Ptr::zero(tape_len)).value();
        let (_, effect) = compile_brainfuck_into(
            include_str!("../../examples/rot13.bf"),
            &mut graph,
            ptr,
            start.effect(),
        );
        let _end = graph.end(effect);
    }

    driver::run_opt_passes(&mut graph, usize::MAX, &settings.pass_config(), None);

    let (program, ir) =
        driver::sequentialize_graph(&settings, &graph, None, PrettyConfig::minimal()).unwrap();
    println!("{ir}");

    let (mut input, mut output, mut tape) = (
        ByteVec::from(input),
        Vec::with_capacity(128),
        vec![0x00; tape_len as usize],
    );
    let jit = Jit::new(&settings, None, None)
        .unwrap()
        .compile(&program)
        .unwrap();

    // Safety: Decidedly not safe in the slightest
    unsafe {
        jit.execute_into(&mut tape, &mut input, &mut output)
            .unwrap();
    }

    assert_eq!(output, expected_output);
}

test_opts! {
    h,
    use_default_passes = true,
    output = b"H",
    |graph, effect, tape_len| {
        let ptr = graph.int(Ptr::zero(tape_len)).value();
        let (_, effect) = compile_brainfuck_into(
            include_str!("../../examples/h.bf"),
            graph,
            ptr,
            effect,
        );

        effect
    },
}

test_opts! {
    report_30k,
    use_default_passes = true,
    tape_len = 30000,
    output = b"#",
    |graph, effect, tape_len| {
        let ptr = graph.int(Ptr::zero(tape_len)).value();
        let (_, effect) = compile_brainfuck_into(
            include_str!("../../examples/report_30k.bf"),
            graph,
            ptr,
            effect,
        );

        effect
    },
}
