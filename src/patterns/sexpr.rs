use std::{iter::Peekable, str::CharIndices};

#[derive(Debug)]
pub enum Sexpr<'a> {
    Atom(&'a str),
    Cons(Vec<Self>),
}

pub fn parse_sexpr(source: &str) -> Sexpr<'_> {
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

            c if c.is_alphabetic() || c == '_' => {
                let start = idx;
                let mut end = idx;

                let mut current = *chars.peek().unwrap();
                while current.1.is_alphabetic() || current.1 == '_' {
                    end = current.0;
                    chars.next().unwrap();
                    current = *chars.peek().unwrap();
                }

                return Sexpr::Atom(&source[start..=end]);
            }

            c if c.is_whitespace() => {}

            unknown => panic!("unknown char: {}", unknown),
        }
    }

    Sexpr::Cons(Vec::new())
}

#[test]
fn sexpr_smoke_test() {
    let source = "
        (* Comment *)
        (cons (cons nil))
    ";

    println!("{:?}", parse_sexpr(source));
}
