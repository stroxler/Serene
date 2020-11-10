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
use crate::errors::Error;
use crate::runtime;
use crate::scope::Scope;
use crate::types::collections;
use crate::types::{BuiltinFunction, Number, Symbol};
use std::fmt;

pub type AST = Vec<Expr>;

pub type PossibleExpr = Result<Expr, Error>;

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Location {
    position: i64,
    file_path: String,
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "'{}:{}'", &self.file_path, &self.position)
    }
}

pub trait Node: fmt::Display {
    fn location(&self) -> Location {
        Location {
            position: 0,
            file_path: "NotImplemented".to_string(),
        }
    }

    fn get_type_str(&self) -> &str {
        "Some type"
    }
}

pub trait Callable {
    fn apply(&self, rt: &runtime::Runtime, scope: &Scope, args: collections::List) -> Expr;
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum Expr {
    Sym(Symbol),
    Str(String),
    //Fn(Function),
    BuiltinFn(BuiltinFunction),
    Num(Number),
    Error(String),
    Cons(collections::List),
    Nil,
    Comment,
    NoMatch,
}

impl Expr {
    pub fn make_list(elements: &[Expr]) -> Expr {
        Expr::Cons(collections::List::new(elements))
    }

    pub fn list_to_cons(l: collections::List) -> Expr {
        Expr::Cons(l)
    }

    pub fn make_empty_list() -> collections::List {
        collections::List::new_empty()
    }

    pub fn make_symbol(v: String, target_ns: Option<String>) -> Expr {
        Expr::Sym(Symbol::new(v, target_ns))
    }

    pub fn make_string(v: String) -> Expr {
        Expr::Str(v)
    }

    pub fn make_number(n: Number) -> Expr {
        Expr::Num(n)
    }

    pub fn get_node(&self) -> Option<&dyn Node> {
        match self {
            Self::Num(x) => Some(x),
            Self::Sym(x) => Some(x),
            Self::Cons(x) => Some(x),
            Self::BuiltinFn(x) => Some(x),
            //Self::Str(x) => x,
            // Self:://Fn(Function),
            //Self::Error(String),
            _ => None,
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get_node() {
            Some(n) => n.fmt(f),
            _ => match self {
                Self::Comment => write!(f, "comment"),
                Self::Error(_) => write!(f, "error"),
                Self::Nil => write!(f, "nil"),
                Self::NoMatch => write!(f, "noMatch"),
                _ => write!(f, "Should Not happen"),
            },
        }
    }
}

impl Node for Expr {
    fn get_type_str(&self) -> &str {
        match self.get_node() {
            Some(x) => x.get_type_str(),
            None => match self {
                Self::Comment => "comment",
                Self::Error(_) => "error",
                Self::Nil => "nil",
                Self::NoMatch => "noMatch",
                _ => {
                    panic!("This shouldn't happen. Checkout `get_node` and `get_type_str` on Expr")
                }
            },
        }
    }
}
