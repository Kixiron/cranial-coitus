use crate::{
    graph::{GammaData, OutputPort, Rvsdg, ThetaData},
    parse::Token,
    values::Ptr,
};

pub fn lower_tokens(
    graph: &mut Rvsdg,
    mut ptr: OutputPort,
    mut effect: OutputPort,
    tokens: &[Token],
    tape_len: u16,
) -> (OutputPort, OutputPort) {
    // FIXME: Byte node
    let (zero, one) = (graph.byte(0).value(), graph.byte(1).value());

    for token in tokens {
        match token {
            Token::IncPtr => ptr = graph.add(ptr, one).value(),
            Token::DecPtr => ptr = graph.sub(ptr, one).value(),

            Token::Inc => {
                // Load the pointed-to cell's current value
                let load = graph.load(ptr, effect);
                effect = load.output_effect();

                // Increment the loaded cell's value
                let inc = graph.add(load.output_value(), one).value();

                // Store the incremented value into the pointed-to cell
                let store = graph.store(ptr, inc, effect);
                effect = store.output_effect();
            }
            Token::Dec => {
                // Load the pointed-to cell's current value
                let load = graph.load(ptr, effect);
                effect = load.output_effect();

                // Decrement the loaded cell's value
                let dec = graph.sub(load.output_value(), one).value();

                // Store the decremented value into the pointed-to cell
                let store = graph.store(ptr, dec, effect);
                effect = store.output_effect();
            }

            Token::Output => {
                // Load the pointed-to cell's current value
                let load = graph.load(ptr, effect);
                effect = load.output_effect();

                // Output the value of the loaded cell
                let output = graph.output(load.output_value(), effect);
                effect = output.output_effect();
            }
            Token::Input => {
                // Get user input
                let input = graph.input(effect);
                effect = input.output_effect();

                // Store the input's result to the currently pointed-to cell
                let store = graph.store(ptr, input.output_value(), effect);
                effect = store.output_effect();
            }

            Token::Loop(body) => {
                // Load the current cell's value
                let load = graph.load(ptr, effect);
                effect = load.output_effect();

                // Compare the cell's value to zero
                let cmp = graph.eq(load.output_value(), zero);

                // Create a gamma node to decide whether or not to drop into the loop
                // Brainfuck loops are equivalent to this general structure:
                //
                // ```rust
                // if *ptr != 0 {
                //     do { ... } while *ptr != 0;
                // }
                // ```
                //
                // So we translate that into our node structure using a gamma
                // node as the outer `if` and a theta as the inner tail controlled loop
                let gamma = graph.gamma(
                    [ptr],
                    effect,
                    cmp.value(),
                    // The truthy branch (`*ptr == 0`) is empty, we skip the loop entirely
                    // if the cell's value is already zero
                    |_graph, effect, inputs| {
                        let ptr = inputs[0];
                        GammaData::new([ptr], effect)
                    },
                    // The falsy branch where `*ptr != 0`, this is where we run the loop's actual body!
                    |graph, mut effect, inputs| {
                        let mut ptr = inputs[0];

                        // Create the inner theta node
                        // FIXME: Pass in invariant constants
                        let theta = graph.theta(
                            [],
                            [ptr],
                            effect,
                            |graph, effect, _invariant_inputs, variant_inputs| {
                                let [ptr]: [OutputPort; 1] = variant_inputs.try_into().unwrap();
                                let (ptr, effect) =
                                    lower_tokens(graph, ptr, effect, body, tape_len);

                                // FIXME: Byte node
                                let zero = graph.int(Ptr::zero(tape_len));
                                let load = graph.load(ptr, effect);
                                let condition = graph.neq(load.output_value(), zero.value());

                                ThetaData::new([ptr], condition.value(), load.output_effect())
                            },
                        );

                        ptr = theta.outputs()[0];
                        effect = theta
                            .output_effect()
                            .expect("all thetas are effectful right now");

                        GammaData::new([ptr], effect)
                    },
                );

                ptr = gamma.outputs()[0];
                effect = gamma
                    .output_effect()
                    .expect("all gammas are effectful right now");
            }
        }
    }

    (ptr, effect)
}
