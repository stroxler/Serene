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
use crate::ast::{Expr, PossibleExpr};
use crate::errors::err;
use crate::reader::read_string;
use crate::runtime::RT;
use crate::scope::Scope;

fn eval_expr(rt: &RT, scope: &Scope, expr: Expr) -> PossibleExpr {
    Ok(expr)
}

pub fn eval(rt: &RT, scope: &Scope, exprs: Vec<Expr>) -> PossibleExpr {
    match exprs.last() {
        Some(e) => Ok(e.clone()),
        _ => Err(err("NotImplemented".to_string())),
    }
}

pub fn rep(rt: &RT, scope: &Scope, input: &str) {
    match read_string(input) {
        Ok(exprs) => {
            let result_expr = eval(rt, scope, exprs);
            println!("<<");
        }
        Err(e) => println!("Error: {}", e),
    }
}
