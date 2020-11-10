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
use crate::ast::{Callable, Expr, Node};
use crate::runtime;
use crate::scope::Scope;
use crate::types::collections::List;

use std::fmt;

pub type BuiltinHandler = fn(rt: &runtime::Runtime, scope: &Scope, args: List) -> Expr;

#[derive(Clone)]
pub struct BuiltinFunction {
    pub name: String,
    ns: String,
    handler_function: BuiltinHandler,
}

impl PartialEq for BuiltinFunction {
    fn eq(&self, other: &Self) -> bool {
        (self.name == other.name) && (self.ns == other.ns)
    }
}

impl Eq for BuiltinFunction {}

impl Node for BuiltinFunction {
    fn get_type_str(&self) -> &str {
        "Function"
    }
}

impl Callable for BuiltinFunction {
    fn apply(&self, rt: &runtime::Runtime, scope: &Scope, args: List) -> Expr {
        let f = self.handler_function;
        f(rt, scope, args)
    }
}

impl fmt::Display for BuiltinFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", &self.ns, &self.name)
    }
}

impl fmt::Debug for BuiltinFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuiltinFunction")
            .field("name", &self.name)
            .field("ns", &self.ns)
            .finish()
    }
}
