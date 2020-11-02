/* Serene --- Yet an other Lisp
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
use crate::ast::{Expr, Expression, PossibleExpr, StringRepr};
use crate::errors::err;
use crate::reader::read_string;
use crate::runtime::RT;
use crate::scope::Scope;

fn eval_expr(rt: &RT, scope: &Scope, expr: Expr) -> PossibleExpr {
    match expr {
        Expr::Num(n) => n.eval(rt, scope),
        // TODO: find a better way to attach the ns name to the symbol. This
        //       is ugly.
        Expr::Sym(s) => s
            .clone_with_ns(rt.current_ns().name.clone())
            .eval(rt, scope),
        _ => Ok(expr),
    }
}

pub fn eval(rt: &RT, exprs: Vec<Expr>) -> PossibleExpr {
    if exprs.len() == 0 {
        return Ok(Expr::NoMatch);
    }

    let mut ret: PossibleExpr = Ok(Expr::NoMatch);

    for expr in exprs.iter() {
        ret = eval_expr(rt, rt.current_scope(), expr.clone());
    }

    ret
}

pub fn read_eval_print(rt: &RT, input: &str) {
    match read_string(input) {
        Ok(exprs) => {
            if rt.is_debug() {
                println!("Read Result: \n{:?}\n", exprs);
            }

            let result_expr = eval(rt, exprs);

            if rt.is_debug() {
                println!("Eval Result: \n{:?}\n", result_expr);
            }

            match result_expr {
                Ok(expr) => println!("{}", expr.string_repr(rt)),
                Err(e) => println!("{}", e),
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}
