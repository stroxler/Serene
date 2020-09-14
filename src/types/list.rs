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
use crate::ast::Expr;
use crate::compiler::Compiler;
use crate::types::core::{ExprResult, Expression};

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct List {
    first: Expr,
    rest: Expr,
    length: u64,
}

impl List {
    pub fn new(first: Expr, rest: Expr) -> List {
        // The order of field definition is important here.
        // If we move the `length` after `rest` we're going
        // to break the ownership rule of rust because `rest: rest`
        // is going to move it to the new list and we can not
        // borrow it afterward.
        List {
            length: match &rest {
                Expr::Cons(v) => v.length + 1,
                _ => {
                    if let Expr::Nil = first {
                        0
                    } else {
                        1
                    }
                }
            },
            first,
            rest,
        }
    }
}

impl Expression for List {
    fn eval() {}
    fn code_gen<'ctx>(&self, compiler: &'ctx Compiler) -> ExprResult<'ctx> {
        Err("Not implemented on list".to_string())
    }
}
