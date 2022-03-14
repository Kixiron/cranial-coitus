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
