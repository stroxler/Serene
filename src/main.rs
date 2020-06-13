#[macro_use]
extern crate lalrpop_util;
lalrpop_mod!(pub grammer);

pub mod ast;
pub mod collections;


use collections::list;

#[test]
fn grammer() {
    assert!(grammer::ExprsParser::new().parse("a22").is_ok());
    assert!(grammer::ExprsParser::new().parse("44").is_ok());
    assert!(grammer::ExprsParser::new().parse("is-ok").is_ok());
    assert!(grammer::ExprsParser::new().parse("is-ok?").is_ok());
    assert!(grammer::ExprsParser::new().parse("is-ok+<<>_").is_ok());
    println!("{:?}", grammer::ExprsParser::new().parse("(as
(sad asd) 3i)"));
    assert!(true);
    assert!(grammer::ExprsParser::new()
        .parse("(\"asd\" (symbol (n)))")
        .is_ok());
    assert!(grammer::ExprsParser::new()
        .parse("(\"asd\" (symbol (n 32)))")
        .is_ok());
    assert!(grammer::ExprsParser::new().parse("((22)").is_err());
}

fn main() {
    list::List::h();
}
