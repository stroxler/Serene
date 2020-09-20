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
use crate::types::collections::core::Seq;
use crate::types::{ExprResult, Expression, List};
use crate::values::Value;

pub fn def<'ctx, 'val: 'ctx>(compiler: &'ctx mut Compiler<'val>, args: List) -> ExprResult<'val> {
    // TODO: We need to support docstrings for def
    if args.length() != 2 {
        // TODO: Raise a meaningful error by including the location
        panic!(format!(
            "`def` expects 2 parameters, '{}' given.",
            args.length()
        ));
    }

    let sym = match args.first() {
        Some(e) => match e {
            Expr::Sym(e) => e,
            _ => return Err("First argument of 'def' has to be a symbol".to_string()),
        },
        _ => return Err("First argument of 'def' has to be a symbol".to_string()),
    };

    let value = match args.rest().first() {
        Some(e) => {
            let generated_code = e.code_gen(compiler);
            Value::new(Some(sym.name.clone()), e, generated_code)
        }
        _ => return Err("Missing the second arugment for 'def'.".to_string()),
    };

    compiler
        .current_ns()
        .unwrap()
        .define(sym.name.clone(), value, true)
}
