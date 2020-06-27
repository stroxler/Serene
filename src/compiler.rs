use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::passes::PassManager;
use inkwell::values::{BasicValue, BasicValueEnum, FloatValue, FunctionValue, PointerValue};

use crate::namespace::Namespace;
use crate::types::Expression;
use std::collections::HashMap;

pub struct Compiler<'a, 'ctx> {
    pub context: &'ctx Context,
    pub builder: &'a Builder<'ctx>,
    /// This hashmap contains all the namespaces that has to be compiled and
    /// maps two different keys to the same namespace. Since namespace names
    /// can not contain `/` char, the keys of this map are the namespace
    /// name and the path to the file containing the namespace. For example:
    ///
    /// A let's say we have a namespace `abc.xyz`, this namespace will have
    /// two entries in this hashmap. One would be the ns name itself which
    /// is `abc.xyz` in this case and the otherone would be
    /// `/path/to/abc/xyz.srn` file that contains the ns.
    pub namespaces: HashMap<String, &'a Namespace<'a, 'ctx>>,
    pub fpm: &'a PassManager<FunctionValue<'ctx>>,

    current_ns_name: Option<&'a str>,
}

impl<'a, 'ctx> Compiler<'a, 'ctx> {
    #[inline]
    pub fn current_ns(&self) -> Option<&'a Namespace<'a, 'ctx>> {
        let ns = self.current_ns_name?;
        self.namespaces.get(ns).map(|x| *x)
    }

    /// Returns the `FunctionValue` representing the function being compiled.
    #[inline]
    pub fn current_fn(&self) -> FunctionValue<'ctx> {
        self.current_ns().unwrap().current_fn()
    }

    /// Creates a new stack allocation instruction in the entry block of the function.
    fn create_entry_block_alloca(&self, name: &str) -> PointerValue<'ctx> {
        let builder = self.context.create_builder();

        let entry = self.current_fn().get_first_basic_block().unwrap();

        match entry.get_first_instruction() {
            Some(first_instr) => builder.position_before(&first_instr),
            None => builder.position_at_end(entry),
        }

        builder.build_alloca(self.context.f64_type(), name)
    }

    pub fn compile(exprs: &impl Expression) {}
}
