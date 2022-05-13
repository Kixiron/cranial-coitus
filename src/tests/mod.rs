#![cfg(test)]

test_opts! {
    beer,
    use_default_passes = true,
    step_limit = usize::MAX,
    output = include_bytes!("corpus/beer.out"),
    |graph, effect, tape_len| {
        let ptr = graph.int(Ptr::zero(tape_len)).value();
        let (_, effect) = compile_brainfuck_into(
            include_str!("corpus/beer.bf"),
            graph,
            ptr,
            effect,
        );

        effect
    },
}
