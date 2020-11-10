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
use crate::ast::{Callable, Expr, Node, PossibleExpr};
use crate::errors::err;
use crate::namespace;
use crate::reader::read_string;
use crate::runtime;
use crate::scope::Scope;
use crate::types::collections::Seq;

pub fn string_repr(rt: &runtime::Runtime, expr: &Expr) -> String {
    match expr {
        Expr::Sym(s) => format!(
            "#'{}/{}",
            namespace::get_name(runtime::current_ns(rt)),
            &s.name
        ),
        _ => format!("{}", expr),
    }
}

pub fn eval_expr(rt: &runtime::Runtime, scope: &Scope, expr: Expr) -> PossibleExpr {
    match &expr {
        // ** Number evaluation
        Expr::Num(_) => Ok(expr),

        // ** Symbol evaluation
        Expr::Sym(s) => {
            if s.is_ns_qualified() {
                return namespace::lookup_external(
                    runtime::current_ns(rt),
                    &s.target_ns.clone().unwrap(),
                    &s.name,
                );
            }

            match scope.lookup(&s.name) {
                Some(e) => Ok(e.expr.clone()),
                _ => Err(err(format!(
                    "Undefined binding {} in ns '{}' at {}",
                    string_repr(rt, &expr),
                    namespace::get_name(runtime::current_ns(rt)),
                    s.location()
                ))),
            }
        }

        // ** List evaluation
        Expr::Cons(l) => {
            if l.count() == 0 {
                return Ok(Expr::Nil);
            }

            let first = l.first().unwrap();
            let rest = l.rest();
            match eval_expr(rt, scope, first) {
                Ok(e) => match e {
                    //Expr::Fn(f) => f.apply(rt, scope, rest),
                    Expr::BuiltinFn(b) => Ok(b.apply(rt, scope, rest)),
                    _ => Err(err(format!(
                        "Can't cast '{}' to functions at {}",
                        e.get_type_str(),
                        l.location(),
                    ))),
                },
                Err(e) => Err(e),
            }
        }
        _ => Ok(expr),
    }
}

pub fn eval(rt: &runtime::Runtime, scope: &Scope, exprs: Vec<Expr>) -> PossibleExpr {
    if exprs.len() == 0 {
        return Ok(Expr::NoMatch);
    }

    let mut ret: PossibleExpr = Ok(Expr::NoMatch);

    for expr in exprs.iter() {
        ret = eval_expr(rt, scope, expr.clone());
    }

    ret
}

pub fn read_eval_print(rt: &runtime::Runtime, input: &str) {
    match read_string(input) {
        Ok(exprs) => {
            if runtime::is_debug(rt) {
                println!("Read Result: \n{:?}\n", exprs);
            }

            let result_expr = eval(
                rt,
                namespace::get_root_scope(&runtime::current_ns(rt)),
                exprs,
            );

            if runtime::is_debug(rt) {
                println!("Eval Result: \n{:?}\n", result_expr);
            }

            match result_expr {
                Ok(expr) => println!("{}", string_repr(rt, &expr)),
                Err(e) => println!("{}", e),
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}
