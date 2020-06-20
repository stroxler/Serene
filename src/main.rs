extern crate llvm_sys;

use clap::{load_yaml, App};
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::string::String;

pub mod ast;
pub mod collections;
pub mod expr;
pub mod reader;

fn main() -> io::Result<()> {
    let yaml = load_yaml!("cli.yml");
    let args = App::from(yaml).get_matches();

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
