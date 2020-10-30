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
use clap::{load_yaml, App};
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::string::String;

pub mod ast;
pub mod builtins;
pub mod compiler;
pub mod namespace;
pub mod reader;
pub mod scope;
pub mod types;
pub mod values;

fn main() -> io::Result<()> {
    let yaml = load_yaml!("cli.yml");
    let args = App::from(yaml).get_matches();

    // Create a compiler
    let mut compiler = compiler::Compiler::new();

    compiler.create_ns("user".to_string(), None);
    compiler.set_current_ns("user".to_string());

    if let Some(input) = args.value_of("INPUT") {
        let mut f = File::open(input)?;

        let mut buf = String::new();
        f.read_to_string(&mut buf)?;
        match reader::read_string(&buf) {
            Ok(v) => {
                println!("AST: {:#?}", v);
            }
            Err(e) => println!(">> error {:?}", e),
        }
    } else {
        println!("Input file is missing.")
    }

    Ok(())
}
