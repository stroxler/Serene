use crate::types::Expression;

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

impl Expression for Symbol {
    fn eval() {}
    fn code_gen() {}
}

impl Symbol {
    pub fn new(name: String) -> Self {
        Symbol { name }
    }
}
