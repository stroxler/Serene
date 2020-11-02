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
use crate::namespace::Namespace;
use crate::scope::Scope;
use std::collections::HashMap;
use std::env;
use std::path::Path;

const SERENE_HISTORY_FILE: &'static str = ".serene.hitory";

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
    pub namespaces: HashMap<String, Namespace>,
    current_ns_name: Option<String>,
    debug: bool,
}

impl RT {
    pub fn new() -> RT {
        RT {
            namespaces: HashMap::new(),
            current_ns_name: None,
            debug: false,
        }
    }

    /// Create a new namespace with the given `ns_name` and add it to the current
    /// runtime.
    pub fn create_ns(&mut self, ns_name: String, source_file: Option<String>) {
        self.namespaces
            .insert(ns_name.clone(), Namespace::new(ns_name, source_file));
    }

    /// Set the current ns to the given `ns_name`. The `ns_name` has to be
    /// part of the runtime already.
    pub fn set_current_ns(&mut self, ns_name: String) {
        match self.namespaces.get(&ns_name) {
            Some(_) => self.current_ns_name = Some(ns_name),
            None => panic!("The given namespace '{}' doesn't exit", ns_name),
        }
    }

    pub fn current_ns(&self) -> &Namespace {
        if let None = self.current_ns_name {
            // `current_ns_name` has to be not None all the time.
            panic!("No namespace has been set to current.");
        }

        self.namespaces
            .get(&self.current_ns_name.clone().unwrap())
            .unwrap()
    }

    pub fn current_scope(&self) -> &Scope {
        self.current_ns().current_scope()
    }

    #[inline]
    pub fn set_debug_mode(&mut self, v: bool) {
        self.debug = v;
    }

    #[inline]
    pub fn is_debug(&self) -> bool {
        self.debug
    }

    pub fn history_file_path(&self) -> String {
        match env::var("HOME") {
            Ok(v) => {
                let history = Path::new(&v).join(SERENE_HISTORY_FILE).clone();
                history.to_str().unwrap().into()
            }
            Err(_) => SERENE_HISTORY_FILE.into(),
        }
    }
}
