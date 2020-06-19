extern crate llvm_sys;

use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::string::String;

pub mod ast;
pub mod collections;
pub mod reader;

fn main() -> io::Result<()> {
    // let input = String::from("(println \">>>>>\" '(+ 2 -3))");

    // println!("{:?}", reader::read_string(&input).unwrap());
    let mut f = File::open(
        "/home/lxsameer/src/serene/serene/resources/benchmarks/parsers/example_code.srn",
    )?;

    let mut buf = String::new();
    f.read_to_string(&mut buf)?;
    match reader::read_string(&buf) {
        Ok(v) => println!("{:?}", v),
        Err(e) => println!(">> error {:?}", e),
    }
    Ok(())
}
