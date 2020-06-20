use crate::collections::list;
use crate::expr::Expression;

// Note: I kept the number implementation simple for now
// but we need to decide on our approach to numbers, are
// we going to only support the 64bit variants? or should
// try to be smart and support 32 and 64 and switch between
// them ?
// What about usize and isize ?
#[derive(Debug, Clone)]
pub enum Number {
    Integer(i64),
    Float(f64),
}

impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        let comb = (&self, &other);

        match comb {
            (Number::Integer(x), Number::Integer(y)) => *x == *y,
            (Number::Float(x), Number::Float(y)) => *x == *y,
            (Number::Integer(x), Number::Float(y)) => *x as f64 == *y,
            (Number::Float(x), Number::Integer(y)) => *x == *y as f64,
        }
    }
}

impl Eq for Number {}

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum Expr {
    //List(list::List<Expr>),
    Symbol(String),
    Str(String),
    Quote(Box<Expr>),
    Num(Number),
    Comment,
    Error(String),
    //    Cons(Box<Expr>, Box<Expr>),
    Cons(list::List<Expr>),
    Nil,
    NoMatch,
}

impl Expr {
    pub fn make_list(first: Expr, rest: Expr) -> Expr {
        Expr::Cons(list::List::<Expr>::new(Box::new(first), Box::new(rest)))
    }
}

impl Expression for Expr {
    fn eval() {}
    fn code_gen() {}
}
