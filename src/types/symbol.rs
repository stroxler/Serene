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
use crate::compiler::Compiler;
use crate::types::core::{ExprResult, Expression};
use inkwell::types::IntType;
use inkwell::values::AnyValueEnum;

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
}

impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Symbol {}

impl Expression for Symbol {
    fn eval() {}
    fn code_gen<'ctx, 'val: 'ctx>(&self, compiler: &'ctx mut Compiler<'val>) -> ExprResult<'val> {
        let bool_t: IntType<'val> = compiler.context.bool_type();
        if self.name == "true" {
            Ok(AnyValueEnum::IntValue(bool_t.const_int(1, false)))
        } else if self.name == "false" {
            Ok(AnyValueEnum::IntValue(bool_t.const_int(0, false)))
        } else {
            Err("Nothing yet".to_string())
        }
    }
}

impl Symbol {
    pub fn new(name: String) -> Self {
        Symbol { name }
    }

    pub fn is_def(&self) -> bool {
        self.name == "def"
    }
}
