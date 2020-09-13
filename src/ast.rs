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
use crate::types::{ExprResult, Expression, List, Number, Symbol};

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum Expr {
    Sym(Symbol),
    Str(String),
    Num(Number),
    Comment,
    Error(String),
    Cons(List<Expr>),
    Nil,
    NoMatch,
}

impl Expr {
    pub fn make_list(first: Expr, rest: Expr) -> Expr {
        Expr::Cons(List::<Expr>::new(Box::new(first), Box::new(rest)))
    }

    pub fn make_symbol(v: String) -> Expr {
        Expr::Sym(Symbol::new(v))
    }

    pub fn make_string(v: String) -> Expr {
        Expr::Str(v)
    }

    pub fn make_number(n: Number) -> Expr {
        Expr::Num(n)
    }
}

impl Expression for Expr {
    fn eval() {}
    fn code_gen<'ctx>(&self, compiler: &'ctx Compiler) -> ExprResult<'ctx> {
        match self {
            Expr::Sym(s) => s.code_gen(compiler),
            _ => Err("NotImplemented".to_string()),
        }
    }
}
