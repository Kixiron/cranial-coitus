use std::cmp::max;

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

#[derive(Debug)]
pub struct Parsed {
    pub tokens: Box<[Token]>,
    pub source_len: usize,
    pub total_tokens: usize,
    pub deepest_nesting: usize,
}

pub fn parse(source: &str) -> Parsed {
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

    let (mut scopes, mut total_tokens, mut deepest_nesting) = (vec![Vec::new()], 0, 0);
    for token in tokens {
        match token {
            RawToken::IncPtr => scopes.last_mut().unwrap().push(Token::IncPtr),
            RawToken::DecPtr => scopes.last_mut().unwrap().push(Token::DecPtr),
            RawToken::Inc => scopes.last_mut().unwrap().push(Token::Inc),
            RawToken::Dec => scopes.last_mut().unwrap().push(Token::Dec),
            RawToken::Output => scopes.last_mut().unwrap().push(Token::Output),
            RawToken::Input => scopes.last_mut().unwrap().push(Token::Input),
            RawToken::JumpStart => {
                scopes.push(Vec::new());
                deepest_nesting = max(deepest_nesting, scopes.len());
            }
            RawToken::JumpEnd => {
                let body = scopes.pop().unwrap();
                scopes
                    .last_mut()
                    .unwrap()
                    .push(Token::Loop(body.into_boxed_slice()));
            }
        }

        total_tokens += 1;
    }

    assert_eq!(scopes.len(), 1);
    let tokens = scopes.remove(0).into_boxed_slice();

    Parsed {
        tokens,
        source_len: source.len(),
        total_tokens,
        deepest_nesting,
    }
}
