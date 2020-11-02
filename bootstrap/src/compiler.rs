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
use crate::namespace::Namespace;
use crate::types::Expression;
use std::collections::HashMap;

pub struct Compiler {
    //pub fpm: &'a PassManager<FunctionValue<'ctx>>,
    current_ns_name: Option<String>,
}

impl Compiler {
    pub fn new() -> Compiler {
        //let user_ns = Namespace::new(&context, default_ns_name);
        //namespaces.insert(default_ns_name, &user_ns);
        // let fpm = PassManager::create(&user_ns.module);

        // fpm.add_instruction_combining_pass();
        // fpm.add_reassociate_pass();
        // fpm.add_gvn_pass();
        // fpm.add_cfg_simplification_pass();
        // fpm.add_basic_alias_analysis_pass();
        // fpm.add_promote_memory_to_register_pass();
        // fpm.add_instruction_combining_pass();
        // fpm.add_reassociate_pass();

        // fpm.initialize();
        Compiler {
            namespaces: HashMap::new(),
            current_ns_name: None,
        }
    }

    pub fn create_ns(&mut self, ns_name: String, source_file: Option<&str>) {
        self.namespaces
            .insert(ns_name.clone(), Namespace::new(ns_name, None));
    }

    pub fn set_current_ns(&mut self, ns_name: String) {
        self.current_ns_name = Some(ns_name);
    }

    #[inline]
    pub fn current_ns(&mut self) -> Option<&mut Namespace> {
        match &self.current_ns_name {
            Some(ns) => self.namespaces.get_mut(ns).map(|x| x),
            _ => None,
        }
    }
}
