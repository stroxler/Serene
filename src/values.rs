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
use crate::types::ExprResult;

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Value<'a> {
    pub llvm_id: Option<String>,
    pub llvm_value: ExprResult<'a>,
    pub expr: Expr,
}

impl<'a> Value<'a> {
    pub fn new(name: Option<String>, expr: Expr, value: ExprResult<'a>) -> Value {
        Value {
            llvm_id: name,
            llvm_value: value,
            expr: expr,
        }
    }
}
