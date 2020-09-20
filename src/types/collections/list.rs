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

    pub fn length(&self) -> usize {
        self.elements.len()
    }
}

impl Expression for List {
    fn eval() {}
    fn code_gen<'ctx, 'val: 'ctx>(&self, compiler: &'ctx mut Compiler<'val>) -> ExprResult<'val> {
        match self.first() {
            Some(e) => match e {
                Expr::Sym(s) if s.is_def() => def(compiler, self.rest()),

                _ => Err("Not implemented on list".to_string()),
            },

            // TODO: We need to return an empty list here
            None => Err("Can't not evaluate empty list".to_string()),
        }
        // def(compiler, self);
        // Err("Not implemented on list".to_string())
    }
}

impl Seq<Expr> for List {
    type Coll = List;

    fn first(&self) -> Option<Expr> {
        match self.elements.first() {
            Some(e) => Some(e.clone()),
            None => None,
        }
    }

    fn rest(&self) -> List {
        if self.length() > 0 {
            List::new(&self.elements[1..])
        } else {
            List::new_empty()
        }
    }
}
