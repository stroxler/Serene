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
use crate::builtins::def;
use crate::compiler::Compiler;
use crate::types::collections::core::Seq;
use crate::types::core::{ExprResult, Expression};

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct List {
    pub car: Expr,
    pub cdr: Expr,
    pub length: u64,
}
//pub enum List<T> { Nil, Cons(T, Box<List<T>>) }

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
            car: first,
            cdr: rest,
        }
    }
}

impl Expression for List {
    fn eval() {}
    fn code_gen<'ctx>(&self, compiler: &'ctx Compiler) -> ExprResult<'ctx> {
        // match &self.car {
        //     Expr::Sym(s) => def(compiler, self),
        //     _ => ,
        // }
        def(compiler, self);
        Err("Not implemented on list".to_string())
    }
}

impl Seq<Expr> for List {
    fn first<'a>(&'a self) -> &'a Expr {
        &self.car
    }

    fn rest<'a>(&'a self) -> Option<&'a List> {
        match &self.cdr {
            Expr::Nil => None,
            Expr::Cons(v) => Some(v),
            _ => panic!("'rest' should not match anything else!"),
        }
    }
}
