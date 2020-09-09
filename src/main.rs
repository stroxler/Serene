extern crate inkwell;

use clap::{load_yaml, App};
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::string::String;

pub mod ast;
pub mod compiler;
pub mod namespace;
pub mod reader;
pub mod scope;
pub mod types;

fn main() -> io::Result<()> {
    let yaml = load_yaml!("cli.yml");
    let args = App::from(yaml).get_matches();
    let context = compiler::create_context();
    let compiler = compiler::Compiler::new(&context);

    if let Some(input) = args.value_of("INPUT") {
        let mut f = File::open(input)?;

        let mut buf = String::new();
        f.read_to_string(&mut buf)?;
        match reader::read_string(&buf) {
            Ok(v) => println!("{:?}", v),
            Err(e) => println!(">> error {:?}", e),
        }
    } else {
        println!("Input file is missing.")
    }
    Ok(())
}
