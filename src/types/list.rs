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
use crate::namespace::Namespace;
use crate::types::core::{ExprResult, Expression};

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct List<T>
where
    for<'a> T: Expression<'a>,
{
    first: Box<T>,
    rest: Box<T>,
}

impl<T> List<T>
where
    for<'a> T: Expression<'a>,
{
    pub fn new<S>(first: Box<S>, rest: Box<S>) -> List<S>
    where
        for<'a> S: Expression<'a>,
    {
        List { first, rest }
    }
}

impl<'a, T> Expression<'a> for List<T>
where
    for<'b> T: Expression<'b>,
{
    fn eval() {}
    fn code_gen(&self, ns: &Namespace) -> ExprResult<'a> {
        Err("Not implemented on list".to_string())
    }
}
