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
//use crate::scope::Scope;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::FunctionValue;

pub struct Namespace<'ctx> {
    /// Each namespace in serene contains it's own LLVM module. You can
    /// think of modules as compilation units. Object files if you prefer.
    /// This way we should be able to hot swap the namespaces.
    pub module: Module<'ctx>,
    //scope: Scope<'ctx>,

    // The option of the current function being compiled
    current_fn_opt: Option<FunctionValue<'ctx>>,
}

impl<'ctx> Namespace<'ctx> {
    pub fn new(context: &'ctx Context, name: &str) -> Namespace<'ctx> {
        Namespace {
            module: context.create_module(&name),
            //scope: Scope::new(None),
            current_fn_opt: None,
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
}
