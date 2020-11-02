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
use crate::ast::{Expr, PossibleExpr};
use crate::scope::Scope;

pub struct Namespace {
    /// Root scope of the namespace
    pub name: String,
    pub source_file: Option<String>,
    root_scope: Scope,
}

impl Namespace {
    pub fn new(name: String, source_file: Option<String>) -> Namespace {
        Namespace {
            name,
            source_file,
            root_scope: Scope::new(None),
        }
    }

    pub fn current_scope(&self) -> &Scope {
        &self.root_scope
    }

    pub fn lookup_external(&self, target: &str, key: &str) -> PossibleExpr {
        Ok(Expr::Nil)
    }
}
