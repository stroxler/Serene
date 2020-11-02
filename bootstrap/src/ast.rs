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
use crate::runtime::RT;
use crate::scope::Scope;
use crate::types::collections;
use crate::types::{Number, Symbol};
use std::fmt;

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

pub trait Expression {
    fn location(&self) -> Location {
        Location {
            position: 0,
            file_path: "NotImplemented".to_string(),
        }
    }

    fn eval(&self, rt: &RT, scope: &Scope) -> PossibleExpr;
}

/// It differs from the `fmt::Display` in the way that anything that
/// we want to show in a repl as the result of an evaluation and needs
/// the runtime details has to implement this trait. But we use the
/// `fmt::Display` as a formatter and in a way that it doesn't need the
/// runtime.
pub trait StringRepr {
    fn string_repr(&self, rt: &RT) -> String;
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum Expr {
    Sym(Symbol),
    Str(String),
    Num(Number),
    Comment,
    Error(String),
    Cons(collections::List),
    Nil,
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
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Num(n) => n.fmt(f),
            Expr::Sym(s) => s.fmt(f),
            _ => write!(f, "NA"),
        }
    }
}

impl StringRepr for Expr {
    fn string_repr(&self, rt: &RT) -> String {
        match self {
            Expr::Num(n) => n.string_repr(rt),
            Expr::Sym(s) => s.string_repr(rt),
            Expr::Cons(c) => c.string_repr(rt),
            _ => "NA".to_string(),
        }
    }
}
