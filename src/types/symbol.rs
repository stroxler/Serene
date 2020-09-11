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
use crate::types::core::{ExprResult, Expression};

#[derive(Debug, Clone)]
pub struct Symbol {
    name: String,
}

impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Symbol {}

impl<'a> Expression<'a> for Symbol {
    fn eval() {}
    fn code_gen(&self, compiler: &Compiler) -> ExprResult<'a> {
        Err("Not implemented on symbol".to_string())
    }
}

impl Symbol {
    pub fn new(name: String) -> Self {
        Symbol { name }
    }
}
