/** Serene --- Yet an other Lisp
*
* Copyright (c) 2020  Sameer Rahmani <lxsameer@gnu.org>
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 2 of the License.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
use crate::scope::Scope;
use crate::types::ExprResult;
use crate::values::Value;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{AnyTypeEnum, BasicType, FunctionType};
use inkwell::values::{AnyValueEnum, FunctionValue};

pub struct Namespace<'ctx> {
    /// Each namespace in serene contains it's own LLVM module. You can
    /// think of modules as compilation units. Object files if you prefer.
    /// This way we should be able to hot swap the namespaces.
    pub module: Module<'ctx>,

    /// Root scope of the namespace
    scope: Scope<'ctx>,

    /// The option of the current function being compiled
    current_fn_opt: Option<FunctionValue<'ctx>>,
    // Current scope of the namespace, for example when we're processing
    // a let form, this field would refer to the scope of that let form.
    //current_scope_opt: Option<Scope<'ctx>>,
}

impl<'ctx> Namespace<'ctx> {
    pub fn new(
        context: &'ctx Context,
        name: String,
        source_file: Option<&'ctx str>,
    ) -> Namespace<'ctx> {
        let module = context.create_module(&name);

        module.set_source_file_name(source_file.unwrap_or(&name));

        Namespace {
            module,
            //scope: Scope::new(None),
            current_fn_opt: None,
            scope: Scope::new(None),
            //current_scope_opt: None,
        }
    }
    /// Get a defined function given its name.
    #[inline]
    pub fn get_function(&self, name: &str) -> Option<FunctionValue<'ctx>> {
        self.module.get_function(name)
    }

    /// Return the `FunctionValue` representing the function being compiled.
    #[inline]
    pub fn current_fn(&self) -> FunctionValue<'ctx> {
        self.current_fn_opt.unwrap()
    }

    fn define_function(
        &self,
        name: String,
        value: Value<'ctx>,
        public: bool,
        f: FunctionType<'ctx>,
    ) -> ExprResult<'ctx> {
        Err("NotImpelemnted".to_string())
    }

    fn define_value(
        &mut self,
        name: String,
        value: Value<'ctx>,
        public: bool,
        t: impl BasicType<'ctx>,
    ) -> ExprResult<'ctx> {
        let c = self.module.add_global(t, None, &name);
        match value.llvm_value {
            Ok(v) => {
                match v {
                    AnyValueEnum::ArrayValue(a) => c.set_initializer(&a),
                    AnyValueEnum::IntValue(i) => c.set_initializer(&i),
                    AnyValueEnum::FloatValue(f) => c.set_initializer(&f),
                    AnyValueEnum::PointerValue(p) => c.set_initializer(&p),
                    AnyValueEnum::StructValue(s) => c.set_initializer(&s),
                    AnyValueEnum::VectorValue(v) => c.set_initializer(&v),
                    _ => panic!("It shoudn't happen!!!"),
                };

                self.scope.insert(&name, value, public);
                Ok(v)
            }

            Err(e) => Err(e),
        }
    }

    pub fn define(&mut self, name: String, value: Value<'ctx>, public: bool) -> ExprResult<'ctx> {
        match value.llvm_value {
            Ok(r) => match r.get_type() {
                AnyTypeEnum::FunctionType(f) => self.define_function(name, value, public, f),
                AnyTypeEnum::IntType(i) => self.define_value(name, value, public, i),
                AnyTypeEnum::ArrayType(a) => self.define_value(name, value, public, a),
                AnyTypeEnum::FloatType(f) => self.define_value(name, value, public, f),
                AnyTypeEnum::PointerType(p) => self.define_value(name, value, public, p),
                AnyTypeEnum::StructType(s) => self.define_value(name, value, public, s),
                AnyTypeEnum::VectorType(v) => self.define_value(name, value, public, v),
                _ => Err(format!("Data type '{:?}' is not supported", r.get_type())),
            },
            Err(e) => Err(e),
        }
    }
}
