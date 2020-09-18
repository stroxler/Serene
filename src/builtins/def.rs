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
use crate::types::collections::core::{first, rest};
use crate::types::{ExprResult, List};

pub fn def<'a>(compiler: &'a Compiler, args: &'a List) -> ExprResult<'a> {
    // TODO: We need to support docstrings for def
    // if args.length != 3 {
    //     // TODO: Raise a meaningful error by including the location
    //     panic!(format!(
    //         "`def` expects 2 parameters, '{}' given.",
    //         args.length
    //     ));
    // }

    // //let def_ = &args.first;1
    // let name = first(rest(args));
    // //let value = first(rest(rest(args)));

    // println!("<<<< {:?}", name);
    // // TODO: make sure that `def_` is a symbol and its name is "def"

    Err("Is not completed".to_string())
}
