use std::io::{self, Write};

#[derive(Debug)]
pub enum Token {
    IncPtr,
    DecPtr,
    Inc,
    Dec,
    Output,
    Input,
    Loop(Box<[Self]>),
}

struct CloseLoop;

impl Token {
    pub fn debug_tokens(tokens: &[Self], mut output: impl Write) {
        let mut stack = tokens.iter().map(|token| (Ok(token), 0)).collect();
        Self::debug_tokens_inner(&mut stack, &mut output).expect("failed to debug tokens");
    }

    fn debug_tokens_inner(
        stack: &mut Vec<(Result<&Token, CloseLoop>, usize)>,
        output: &mut dyn Write,
    ) -> io::Result<()> {
        while let Some((token, level)) = stack.pop() {
            // Write the leading padding
            for _ in 0..level * 2 {
                output.write_all(b" ")?;
            }

            match token {
                Ok(token) => match token {
                    Token::Loop(body) => {
                        // Start loops
                        writeln!(output, "Loop {{")?;

                        // Push all tokens from the loop onto the stack in reverse order
                        stack.reserve(body.len() + 1);
                        stack.push((Err(CloseLoop), level));
                        stack.extend(body.iter().rev().map(|token| (Ok(token), level + 1)));
                    }

                    // All other token kinds just get print out alone
                    other => writeln!(output, "{:?}", other)?,
                },

                // Close loops
                Err(CloseLoop) => writeln!(output, "}}")?,
            }
        }

        Ok(())
    }
}

enum RawToken {
    IncPtr,
    DecPtr,
    Inc,
    Dec,
    Output,
    Input,
    JumpStart,
    JumpEnd,
}

pub fn parse(source: &str) -> Box<[Token]> {
    let tokens = source.chars().flat_map(|token| {
        Some(match token {
            '>' => RawToken::IncPtr,
            '<' => RawToken::DecPtr,
            '+' => RawToken::Inc,
            '-' => RawToken::Dec,
            '.' => RawToken::Output,
            ',' => RawToken::Input,
            '[' => RawToken::JumpStart,
            ']' => RawToken::JumpEnd,
            _ => return None,
        })
    });

    let mut scopes = vec![Vec::new()];
    for token in tokens {
        match token {
            RawToken::IncPtr => scopes.last_mut().unwrap().push(Token::IncPtr),
            RawToken::DecPtr => scopes.last_mut().unwrap().push(Token::DecPtr),
            RawToken::Inc => scopes.last_mut().unwrap().push(Token::Inc),
            RawToken::Dec => scopes.last_mut().unwrap().push(Token::Dec),
            RawToken::Output => scopes.last_mut().unwrap().push(Token::Output),
            RawToken::Input => scopes.last_mut().unwrap().push(Token::Input),
            RawToken::JumpStart => scopes.push(Vec::new()),
            RawToken::JumpEnd => {
                let body = scopes.pop().unwrap();
                scopes
                    .last_mut()
                    .unwrap()
                    .push(Token::Loop(body.into_boxed_slice()));
            }
        }
    }

    assert_eq!(scopes.len(), 1);
    scopes.remove(0).into_boxed_slice()
}
