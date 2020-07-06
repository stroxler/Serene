use crate::namespace::Namespace;
use crate::types::core::{ExprResult, Expression};
use inkwell::values::PointerValue;

#[derive(Debug, Clone)]
pub struct Symbol {
    name: String,
}

impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Symbol {}

impl<'a> Expression<'a> for Symbol {
    fn eval() {}
    fn code_gen(&self, ns: &Namespace) -> ExprResult<'a> {
        Err("Not implemented on symbol".to_string())
    }
}

impl Symbol {
    pub fn new(name: String) -> Self {
        Symbol { name }
    }
}
