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
    elements: Vec<Expr>,
}

impl List {
    pub fn new_empty() -> List {
        List { elements: vec![] }
    }

    pub fn new(elems: &[Expr]) -> List {
        List {
            elements: elems.to_vec(),
        }
    }

    pub fn push(&mut self, elem: Expr) {
        self.elements.push(elem)
    }

    // pub fn new(first: T, rest: List<T>) -> List<T> {
    //     List::Cons(first, Box::new(rest))
    // }
}

impl Expression for List {
    fn eval() {}
    fn code_gen<'ctx>(&self, compiler: &'ctx Compiler) -> ExprResult<'ctx> {
        // match &self.car {
        //     Expr::Sym(s) => def(compiler, self),
        //     _ => ,
        // }
        //def(compiler, self);
        Err("Not implemented on list".to_string())
    }
}

// impl Seq<Expr> for List<T> {
//     fn first<'a>(&'a self) -> &'a Expr {
//         &self.car
//     }

//     fn rest<'a>(&'a self) -> Option<&'a List> {
//         match &self.cdr {
//             Expr::Nil => None,
//             Expr::Cons(v) => Some(v),
//             _ => panic!("'rest' should not match anything else!"),
//         }
//     }
// }
