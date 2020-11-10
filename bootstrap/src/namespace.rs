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
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone)]
pub struct Namespace {
    /// Root scope of the namespace
    pub name: String,
    pub source_file: Option<String>,
    pub root_scope: Scope,
}

impl Namespace {
    pub fn new(name: String, source_file: Option<String>) -> Namespace {
        Namespace {
            name,
            source_file,
            root_scope: Scope::new(None),
        }
    }

    pub fn lookup_external(&self, target: &str, key: &str) -> PossibleExpr {
        Ok(Expr::Nil)
    }

    pub fn define_global(&mut self, name: &str, expr: Expr, public: bool) {
        self.root_scope.insert(&name, expr, public);
    }
}

pub fn define_global(ns: &RwLock<Namespace>, name: &str, expr: Expr, public: bool) {
    match ns.write() {
        Ok(mut n) => n.define_global(name, expr, public),
        Err(_) => panic!("Poisoned write lock while defining a global: '{:?}'", ns),
    }
}

pub fn lookup_external(ns: &RwLock<Namespace>, target: &str, key: &str) -> PossibleExpr {
    match ns.read() {
        Ok(n) => n.lookup_external(target, key),
        Err(_) => panic!("Poisoned write lock while defining a global: '{:?}'", ns),
    }
}

pub fn get_name(ns: &RwLock<Namespace>) -> String {
    match ns.read() {
        Ok(n) => n.name.clone(),
        Err(_) => panic!("Poisoned write lock while defining a global: '{:?}'", ns),
    }
}

pub fn get_root_scope(ns: &RwLock<Namespace>) -> &Scope {
    match ns.read() {
        Ok(n) => &n.root_scope,
        Err(_) => panic!(
            "Poisoned write lock while getting the root scope: '{:?}'",
            ns
        ),
    }
}

#[test]
fn test_ns_define_global() {
    let mut ns = Namespace::new("ns1".to_string(), None);
    assert_eq!(ns.root_scope.lookup("blah").is_none(), true);

    ns.define_global("something", Expr::Nil, true);
    let result = ns.root_scope.lookup("something").unwrap();
    assert_eq!(result.expr, Expr::Nil);
    assert_eq!(result.public, true);
}
