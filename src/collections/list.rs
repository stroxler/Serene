use crate::expr::Expression;

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct List<T: Expression> {
    first: Box<T>,
    rest: Box<T>,
}

impl<T: Expression> List<T> {
    pub fn new<S: Expression>(first: Box<S>, rest: Box<S>) -> List<S> {
        List { first, rest }
    }
}

impl<T: Expression> Expression for List<T> {
    fn eval() {}
    fn code_gen() {}
}
