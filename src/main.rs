extern crate llvm_sys;

use std::string::String;

pub mod ast;
pub mod collections;
pub mod reader;

fn main() {
    let input = String::from("(println \">>>>>\" '(+ 2 -3))");

    println!("{:?}", reader::read_string(&input).unwrap());
}
