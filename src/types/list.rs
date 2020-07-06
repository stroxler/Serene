use crate::namespace::Namespace;
use crate::types::core::{ExprResult, Expression};

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct List<T>
where
    for<'a> T: Expression<'a>,
{
    first: Box<T>,
    rest: Box<T>,
}

impl<T> List<T>
where
    for<'a> T: Expression<'a>,
{
    pub fn new<S>(first: Box<S>, rest: Box<S>) -> List<S>
    where
        for<'a> S: Expression<'a>,
    {
        List { first, rest }
    }
}

impl<'a, T> Expression<'a> for List<T>
where
    for<'b> T: Expression<'b>,
{
    fn eval() {}
    fn code_gen(&self, ns: &Namespace) -> ExprResult<'a> {
        Err("Not implemented on list".to_string())
    }
}
