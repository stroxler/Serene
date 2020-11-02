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
use std::collections::HashMap;

/// This struct describes the values in the scope.
pub struct ScopeElement {
    pub expr: Expr,
    pub public: bool,
}

/// Scopes in **Serene** are simply represented by hashmaps. Each
/// Scope optionally has a parent scope that lookups fallback to
/// if the lookup key is missing from the current scope.
pub struct Scope {
    parent: Option<Box<Scope>>,
    symbol_table: HashMap<String, ScopeElement>,
}

impl Scope {
    pub fn new(_parent: Option<Scope>) -> Scope {
        let p = match _parent {
            Some(x) => Some(Box::new(x)),
            None => None,
        };

        Scope {
            parent: p,
            symbol_table: HashMap::new(),
        }
    }

    /// Lookup the given `key` in the scope and if it is not in the current
    /// scope look it up in the `parent` scope.
    pub fn lookup(&self, key: &str) -> Option<&ScopeElement> {
        if self.symbol_table.contains_key(key) {
            self.symbol_table.get(key)
        } else {
            match &self.parent {
                Some(x) => x.lookup(key),
                None => None,
            }
        }
    }

    pub fn insert(&mut self, key: &str, expr: Expr, public: bool) {
        let v = ScopeElement { public, expr };
        self.symbol_table.insert(key.to_string(), v);
    }
}
