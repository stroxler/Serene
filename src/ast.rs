use crate::types::{Expression, List, Number};

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum Expr {
    Symbol(String),
    Str(String),
    Num(Number),
    Comment,
    Error(String),
    Cons(List<Expr>),
    Nil,
    NoMatch,
}

impl Expr {
    pub fn make_list(first: Expr, rest: Expr) -> Expr {
        Expr::Cons(List::<Expr>::new(Box::new(first), Box::new(rest)))
    }

    pub fn make_symbol(v: String) -> Expr {
        Expr::Symbol(v)
    }

    pub fn make_string(v: String) -> Expr {
        Expr::Str(v)
    }

    pub fn make_number(n: Number) -> Expr {
        Expr::Num(n)
    }
}

impl Expression for Expr {
    fn eval() {}
    fn code_gen() {}
}
