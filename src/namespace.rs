use crate::scope::Scope;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::FunctionValue;

pub struct Namespace<'a, 'ctx> {
    /// Each namespace in serene contains it's own LLVM module. You can
    /// think of modules as compilation units. Object files if you prefer.
    /// This way we should be able to hot swap the namespaces.
    pub module: Module<'ctx>,

    scope: Scope<'a>,

    // The option of the current function being compiled
    current_fn_opt: Option<FunctionValue<'ctx>>,
}

impl<'a, 'ctx> Namespace<'a, 'ctx> {
    pub fn new(context: &'ctx Context, name: &str) -> Namespace<'a, 'ctx> {
        let module = context.create_module(&name);
        Namespace {
            module: module,
            scope: Scope::new(None),
            current_fn_opt: None,
        }
    }
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
