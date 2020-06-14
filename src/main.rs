use std::string::String;

pub mod ast;
pub mod reader;
pub mod collections;



fn main() {
    let input = String::from("(println \">>>>>\" '(+ 2 -3))");

    println!("{:?}",
             reader::read_string(&input).unwrap());
}
