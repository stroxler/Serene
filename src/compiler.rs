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
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::passes::PassManager;
use inkwell::values::{AnyValueEnum, BasicValue, FloatValue, FunctionValue, PointerValue};

use crate::namespace::Namespace;
use crate::types::Expression;
use std::collections::HashMap;

pub struct Compiler<'ctx> {
    pub context: &'ctx Context,
    pub builder: Builder<'ctx>,
    // /// This hashmap contains all the namespaces that has to be compiled and
    // /// maps two different keys to the same namespace. Since namespace names
    // /// can not contain `/` char, the keys of this map are the namespace
    // /// name and the path to the file containing the namespace. For example:
    // ///
    // /// A let's say we have a namespace `abc.xyz`, this namespace will have
    // /// two entries in this hashmap. One would be the ns name itself which
    // /// is `abc.xyz` in this case and the otherone would be
    // /// `/path/to/abc/xyz.srn` file that contains the ns.
    pub namespaces: HashMap<&'ctx str, Namespace<'ctx>>,
    //pub fpm: &'a PassManager<FunctionValue<'ctx>>,
    current_ns_name: Option<&'ctx str>,
}

impl<'ctx> Compiler<'ctx> {
    pub fn new(context: &'ctx Context) -> Compiler<'ctx> {
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
            builder: context.create_builder(),
            context: context,
            namespaces: HashMap::new(),
            current_ns_name: None,
        }
    }

    pub fn create_ns(&mut self, ns_name: &'ctx str) {
        self.namespaces
            .insert(ns_name, Namespace::new(&self.context, ns_name));
    }

    pub fn set_current_ns(&mut self, ns_name: &'ctx str) {
        self.current_ns_name = Some(ns_name);
    }

    #[inline]
    pub fn current_ns(&self) -> Option<&Namespace<'ctx>> {
        let ns = self.current_ns_name?;
        self.namespaces.get(ns).map(|x| x)
    }

    /// Returns the `FunctionValue` representing the function being compiled.
    #[inline]
    pub fn current_fn(&self) -> FunctionValue<'ctx> {
        self.current_ns().unwrap().current_fn()
    }

    // /// Creates a new stack allocation instruction in the entry block of the function.
    // // fn create_entry_block_alloca(&self, name: &str) -> PointerValue<'ctx> {
    // //     let builder = self.context.create_builder();

    // //     let entry = self.current_fn().get_first_basic_block().unwrap();

    // //     match entry.get_first_instruction() {
    // //         Some(first_instr) => builder.position_before(&first_instr),
    // //         None => builder.position_at_end(entry),
    // //     }

    // //     builder.build_alloca(self.context.f64_type(), name)
    // // }
}

pub fn create_context() -> Context {
    return Context::create();
}

pub fn compile<'ctx>(
    compiler: &'ctx Compiler,
    exprs: Vec<impl Expression>,
) -> Vec<Result<AnyValueEnum<'ctx>, String>> {
    match compiler.current_ns() {
        Some(ns) => ns,
        None => panic!("Current namespace is not set."),
    };

    let mut generated_code = vec![];

    for expr in &exprs {
        generated_code.push(expr.code_gen(compiler));
    }

    generated_code
}

// pub fn create_compiler<'ctx>() -> Compiler<'ctx> {
//     let default_ns_name = "user";
//     // let builder = context.create_builder();
//     let context = Context::create();
//     //let user_ns = Namespace::new(&context, default_ns_name);
//     //namespaces.insert(default_ns_name, &user_ns);
//     // let fpm = PassManager::create(&user_ns.module);

//     // fpm.add_instruction_combining_pass();
//     // fpm.add_reassociate_pass();
//     // fpm.add_gvn_pass();
//     // fpm.add_cfg_simplification_pass();
//     // fpm.add_basic_alias_analysis_pass();
//     // fpm.add_promote_memory_to_register_pass();
//     // fpm.add_instruction_combining_pass();
//     // fpm.add_reassociate_pass();

//     // fpm.initialize();
//     //, builder, fpm, namespaces, Some(&default_ns_name)
//     //Compiler::new(context)
//     let builder = context.create_builder();
//     Compiler {
//         builder: builder,
//         context: context,
//         namespaces: HashMap::new(),
//     }
// }
