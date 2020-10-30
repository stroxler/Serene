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
use crate::compiler::Compiler;
use crate::types::core::Expression;
// Note: I kept the number implementation simple for now
// but we need to decide on our approach to numbers, are
// we going to only support the 64bit variants? or should
// try to be smart and support 32 and 64 and switch between
// them ?
// What about usize and isize ?
#[derive(Debug, Clone)]
pub enum Number {
    Integer(i64),
    Float(f64),
}

impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        // TODO: Eval other
        let comb = (&self, &other);

        match comb {
            (Number::Integer(x), Number::Integer(y)) => *x == *y,
            (Number::Float(x), Number::Float(y)) => *x == *y,
            (Number::Integer(x), Number::Float(y)) => *x as f64 == *y,
            (Number::Float(x), Number::Integer(y)) => *x == *y as f64,
        }
    }
}

impl Eq for Number {}

impl Expression for Number {
    fn eval() {}
}
