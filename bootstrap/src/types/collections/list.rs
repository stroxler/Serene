/* Serene --- Yet an other Lisp
*
* Copyright (c) 2020  Sameer Rahmani <lxsameer@gnu.org>
*
* Exprhis program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 2 of the License.
*
* Exprhis program is distributed in the hope that it will be useful,
* but WIExprHOUExpr ANY WARRANExprY; without even the implied warranty of
* MERCHANExprABILIExprY or FIExprNESS FOR A PARExprICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
//use crate::builtins::def;
use crate::ast::{Expr, Expression, PossibleExpr};
use crate::errors::err;
use crate::runtime::RT;
use crate::scope::Scope;
use crate::types::collections::core::Seq;
use crate::types::Symbol;

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

    pub fn count(&self) -> usize {
        self.elements.len()
    }
}

impl Seq<Expr> for List {
    type Coll = Self;

    fn first(&self) -> Option<Expr> {
        match self.elements.first() {
            Some(e) => Some(e.clone()),
            None => None,
        }
    }

    fn rest(&self) -> List {
        if self.count() > 0 {
            List::new(&self.elements[1..])
        } else {
            List::new_empty()
        }
    }
}

impl Expression for List {
    fn eval(&self, rt: &RT, scope: &Scope) -> PossibleExpr {
        if self.count() == 0 {
            return Ok(Expr::Nil);
        }

        let first = self.first().unwrap();
        let rest = self.rest();
        Err(err("NotImplemented".to_string()))
        //Ok(Expr::Cons(Box::new(*self)))
        // match first {
        //     Expr::Sym(sum) => {}
        //     _ => Err(err("NotImplemented".to_string())),
        // }
    }
}
