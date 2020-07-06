use crate::namespace::Namespace;
use inkwell::values::PointerValue;

pub type ExprResult<'a> = Result<PointerValue<'a>, String>;

pub trait Expression<'a> {
    fn eval();
    fn code_gen(&self, ns: &Namespace) -> ExprResult<'a>;
}
