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
use crate::ast::{Expr, Expression, Location, PossibleExpr, StringRepr};
use crate::errors::err;
use crate::runtime::RT;
use crate::scope::Scope;
use std::fmt;

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,

    /// This field holds the ns specifier part of the symbol. For example
    /// in case of `somens/somesym`, this field will hold `somens`.
    target_ns: Option<String>,
    /// Name of the namespace which this symbol is in. It doesn't mean
    /// the namespace which this symbol is defined. For example Let's
    /// say we're in ns A, and there is a sumbol `B/x`. This symbol
    /// refers to the symbol `x` in ns B and it's not the same as
    /// the symbol `x` in ns B. They are two different symbols pointing
    /// to the same value. the `ns` value of the one in ns A would be `A`
    /// and the one in B would be `B`.
    ns: Option<String>,
}

impl Symbol {
    pub fn new(name: String, target_ns: Option<String>) -> Symbol {
        Symbol {
            name,
            target_ns,
            ns: None,
        }
    }

    pub fn is_ns_qualified(&self) -> bool {
        !self.target_ns.is_none()
    }

    pub fn is_def(&self) -> bool {
        self.name == "def"
    }

    /// Only clones the symbol if ns isn't set yet.
    pub fn clone_with_ns(self, ns_name: String) -> Symbol {
        if let Some(_) = self.ns {
            return self;
        }

        Symbol {
            ns: Some(ns_name),
            ..self.clone()
        }
    }
}

impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Symbol {}

impl Expression for Symbol {
    fn eval(&self, rt: &RT, scope: &Scope) -> PossibleExpr {
        if self.is_ns_qualified() {
            return rt
                .current_ns()
                .lookup_external(&self.target_ns.clone().unwrap(), &self.name);
        }

        match scope.lookup(&self.name) {
            Some(e) => Ok(e.expr.clone()),
            _ => Err(err(format!(
                "Undefined binding {} in ns '{}' at {}",
                self,
                self.ns.clone().unwrap(),
                self.location()
            ))),
        }
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.name)
    }
}

impl StringRepr for Symbol {
    fn string_repr(&self, rt: &RT) -> String {
        format!("#'{}/{}", &rt.current_ns().name, &self.name)
    }
}
