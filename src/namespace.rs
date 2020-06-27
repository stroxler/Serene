use inkwell::module::Module;
use inkwell::values::{FunctionValue, PointerValue};
use std::collections::HashMap;

pub struct Namespace<'a, 'ctx> {
    /// Each namespace in serene contains it's own LLVM module. You can
    /// think of modules as compilation units. Object files if you prefer.
    /// This way we should be able to hot swap the namespaces.
    pub module: &'a Module<'ctx>,

    scope: HashMap<String, PointerValue<'ctx>>,

    // The option of the current function being compiled
    current_fn_opt: Option<FunctionValue<'ctx>>,
}

impl<'a, 'ctx> Namespace<'a, 'ctx> {
    /// Gets a defined function given its name.
    #[inline]
    pub fn get_function(&self, name: &str) -> Option<FunctionValue<'ctx>> {
        self.module.get_function(name)
    }

    /// Returns the `FunctionValue` representing the function being compiled.
    #[inline]
    pub fn current_fn(&self) -> FunctionValue<'ctx> {
        self.current_fn_opt.unwrap()
    }
}
