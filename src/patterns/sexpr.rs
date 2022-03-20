use std::{iter::Peekable, str::CharIndices};

#[derive(Debug, PartialEq)]
pub enum Sexpr<'a> {
    Atom(&'a str),
    Cons(Vec<Self>),
}

impl<'a> Sexpr<'a> {
    pub fn parse(source: &'a str) -> Self {
        parse_sexpr(source)
    }

    #[track_caller]
    pub fn to_atom(&self) -> &'a str {
        if let Self::Atom(atom) = self {
            atom
        } else {
            panic!("attempted to get an atom out of {:?}", self);
        }
    }

    #[track_caller]
    pub fn to_cons(&self) -> &[Self] {
        if let Self::Cons(cons) = self {
            cons
        } else {
            panic!("attempted to get a cons out of {:?}", self);
        }
    }
}

fn parse_sexpr(source: &str) -> Sexpr<'_> {
    let mut chars = source.char_indices().peekable();

    parse_sexpr_inner(source, &mut chars)
}

fn parse_sexpr_inner<'a>(source: &'a str, chars: &mut Peekable<CharIndices>) -> Sexpr<'a> {
    while let Some((idx, char)) = chars.next() {
        match char {
            '(' => {
                // Comments
                if chars.peek().unwrap().1 == '*' {
                    chars.next().unwrap();

                    while chars.next().unwrap().1 != '*' && chars.peek().unwrap().1 != ')' {}
                    assert_eq!(chars.next().unwrap().1, ')');

                // Cons
                } else {
                    let mut cons = Vec::new();
                    while chars.peek().unwrap().1 != ')' {
                        cons.push(parse_sexpr_inner(source, chars));

                        while chars.peek().unwrap().1.is_whitespace() {
                            chars.next().unwrap();
                        }
                    }

                    assert_eq!(chars.next().unwrap().1, ')');

                    return Sexpr::Cons(cons);
                }
            }

            c if is_ident(c) => {
                let start = idx;
                let mut end = idx;

                let mut current = *chars.peek().unwrap();
                while is_ident(current.1) {
                    end = current.0;
                    chars.next().unwrap();
                    if let Some(&peek) = chars.peek() {
                        current = peek;
                    } else {
                        break;
                    }
                }

                return Sexpr::Atom(&source[start..=end]);
            }

            c if c.is_whitespace() => {}

            unknown => panic!("unknown char: {}", unknown),
        }
    }

    Sexpr::Cons(Vec::new())
}

fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '?' | '+' | '-' | '*' | '/')
}

#[test]
fn sexpr_smoke_test() {
    let source = "
        (* Comment *)
        (cons (cons nil))
    ";

    println!("{:?}", parse_sexpr(source));
}
