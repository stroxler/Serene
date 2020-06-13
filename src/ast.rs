// Note: I kept the number implementation simple for now
// but we need to decide on our approach to numbers, are
// we going to only support the 64bit variants? or should
// try to be smart and support 32 and 64 and switch between
// them ?
// What about usize and isize ?
#[derive(Debug, Clone)]
pub enum Number {
    Integer(i32),
    Float(f32),
}

impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        let comb = (&self, &other);

        match comb {
            (Number::Integer(x), Number::Integer(y)) => *x == *y,
            (Number::Float(x), Number::Float(y)) => *x == *y,
            (Number::Integer(x), Number::Float(y)) => *x as f32 == *y,
            (Number::Float(x), Number::Integer(y)) => *x == *y as f32,
        }
    }
}

impl Eq for Number {}

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum Expr {
    List(Vec<Expr>),
    Symbol(String),
    Str(String),
    Quote(Box<Expr>),
    Num(Number),
    Comment,
    Error(String),
}
