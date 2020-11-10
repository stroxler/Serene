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
use crate::ast::Expr;
use crate::namespace::Namespace;
use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::sync::{Arc, RwLock};

const SERENE_HISTORY_FILE: &'static str = ".serene.hitory";

pub type Runtime = Arc<RwLock<RT>>;

pub struct RT {
    /// This hashmap contains all the namespaces that has to be compiled and
    /// maps two different keys to the same namespace. Since namespace names
    /// can not contain `/` char, the keys of this map are the namespace
    /// name and the path to the file containing the namespace. For example:
    ///
    /// A let's say we have a namespace `abc.xyz`, this namespace will have
    /// two entries in this hashmap. One would be the ns name itself which
    /// is `abc.xyz` in this case and the otherone would be
    /// `/path/to/abc/xyz.srn` file that contains the ns.
    pub namespaces: HashMap<String, RwLock<Namespace>>,
    pub current_ns_name: Option<String>,
    pub debug: bool,
}

pub fn create_runtime() -> Runtime {
    Arc::new(RwLock::new(RT {
        namespaces: HashMap::new(),
        current_ns_name: None,
        debug: false,
    }))
}

/// Create a new namespace with the given `ns_name` and add it to the current
/// runtime.
pub fn create_ns(rt: &Runtime, ns_name: String, source_file: Option<String>) {
    let mut r = match rt.write() {
        Ok(r) => r,

        Err(_) => panic!("Poisoned runtime!"),
    };

    r.namespaces.insert(
        ns_name.clone(),
        RwLock::new(Namespace::new(ns_name, source_file)),
    );
}

/// Set the current ns to the given `ns_name`. The `ns_name` has to be
/// part of the runtime already.
pub fn set_current_ns(rt: &Runtime, ns_name: String) {
    let mut r = match rt.write() {
        Ok(r) => r,
        Err(_) => panic!("Poisoned runtime!"),
    };

    match r.namespaces.get(&ns_name) {
        Some(_) => r.current_ns_name = Some(ns_name),
        None => panic!("The given namespace '{}' doesn't exit", ns_name),
    }
}

pub fn current_ns(rt: &Runtime) -> &RwLock<Namespace> {
    match rt.read() {
        Ok(mut r) => {
            let ns_name = r.current_ns_name.clone().unwrap();
            match r.namespaces.get(&ns_name) {
                Some(x) => x,
                _ => panic!("No namespace has been set to current."),
            }
        }
        Err(_) => panic!("Poisoned runtime!"),
    }
}

#[inline]
pub fn set_debug_mode(rt: &Runtime, v: bool) {
    let mut r = match rt.write() {
        Ok(r) => r,
        Err(_) => panic!("Poisoned runtime!"),
    };
    r.debug = v;
}

#[inline]
pub fn is_debug(rt: &Runtime) -> bool {
    match rt.read() {
        Ok(r) => r.debug,
        Err(_) => panic!("Poisoned runtime!"),
    }
}

// TODO: Move this function to somewhere else
pub fn history_file_path() -> String {
    match env::var("HOME") {
        Ok(v) => {
            let history = Path::new(&v).join(SERENE_HISTORY_FILE).clone();
            history.to_str().unwrap().into()
        }
        Err(_) => SERENE_HISTORY_FILE.into(),
    }
}

#[test]
fn test_runtime_ns_mutation() {
    let rt = create_runtime();
    create_ns(&rt, "user".to_string(), None);
    set_current_ns(&rt, "user".to_string());
    let ns = current_ns(&rt).read().unwrap();
    assert_eq!(ns.root_scope.lookup("blah").is_none(), true);

    let ns = current_ns(&rt).write().unwrap();
    ns.define_global("something", Expr::Nil, true);
    let result = ns.root_scope.lookup("something").unwrap();
    assert_eq!(result.expr, Expr::Nil);
    assert_eq!(result.public, true);
}
