#![cfg(test)]

test_opts! {
    beer,
    default_passes = true,
    output = include_bytes!("corpus/beer.out"),
    |graph, effect| {
        let ptr = graph.int(0).value();
        let (_, effect) = compile_brainfuck_into(
            include_str!("corpus/beer.bf"),
            graph,
            ptr,
            effect,
        );

        effect
    },
}
