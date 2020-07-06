use inkwell::values::PointerValue;
use std::collections::HashMap;

pub struct Scope<'a> {
    parent: Option<Box<Scope<'a>>>,
    symbol_table: HashMap<String, PointerValue<'a>>,
}

impl<'a> Scope<'a> {
    pub fn new(_parent: Option<Scope>) -> Scope {
        let p = match _parent {
            Some(x) => Some(Box::new(x)),
            None => None,
        };

        Scope {
            parent: p,
            symbol_table: HashMap::new(),
        }
    }

    pub fn lookup(&self, key: &str) -> Option<PointerValue> {
        self.symbol_table.get(key).map(|x| *x)
    }

    pub fn insert(&mut self, key: &str, val: PointerValue<'a>) {
        self.symbol_table.insert(key.to_string(), val);
    }
}
