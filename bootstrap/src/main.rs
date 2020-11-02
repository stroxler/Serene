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
use clap::{load_yaml, App, ArgMatches};
use std::io;

pub mod ast;
pub mod builtins;
pub mod core;
pub mod errors;
pub mod namespace;
pub mod reader;
pub mod repl;
pub mod runtime;
pub mod scope;
pub mod types;

fn repl(args: ArgMatches) {
    let mut debug = false;

    if args.is_present("debug-mode") {
        debug = true;
    }

    let mut rt = runtime::RT::new();

    rt.create_ns("user".to_string(), None);
    rt.set_current_ns("user".to_string());
    rt.set_debug_mode(debug);
    repl::repl(rt);
}

fn main() -> io::Result<()> {
    let yaml = load_yaml!("cli.yml");
    let app = App::from(yaml);

    match app.get_matches().subcommand() {
        Some(("repl", args)) => repl(args.clone()),
        Some(("run", args)) => println!("repl, {:?}", args),
        _ => {}
    }

    // if let Some(input) = args.value_of("INPUT") {
    //     let mut f = File::open(input)?;

    //     let mut buf = String::new();
    //     f.read_to_string(&mut buf)?;
    //     match reader::read_string(&buf) {
    //         Ok(v) => {
    //             println!("AST: {:#?}", v);
    //         }
    //         Err(e) => println!(">> error {:?}", e),
    //     }
    // } else {
    //     println!("Input file is missing.")
    // }

    Ok(())
}
