use std::io::Write;

#[derive(Debug)]
pub enum Token {
    IncPtr,
    DecPtr,
    Inc,
    Dec,
    Output,
    Input,
    Loop(Vec<Self>),
}

impl Token {
    pub fn debug_tokens(tokens: &[Self], mut output: impl Write) {
        Self::debug_tokens_inner(tokens, &mut output, 0);
    }

    fn debug_tokens_inner(tokens: &[Self], output: &mut dyn Write, level: usize) {
        let padding = " ".repeat(level * 2);

        for token in tokens {
            match token {
                Token::Loop(body) => {
                    writeln!(output, "{}Loop {{", padding).unwrap();
                    Self::debug_tokens_inner(body, output, level + 1);
                    writeln!(output, "{}}}", padding).unwrap();
                }

                other => writeln!(output, "{}{:?}", padding, other).unwrap(),
            }
        }
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

pub fn parse(source: &str) -> Vec<Token> {
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
                scopes.last_mut().unwrap().push(Token::Loop(body));
            }
        }
    }

    assert_eq!(scopes.len(), 1);
    scopes.remove(0)
}
