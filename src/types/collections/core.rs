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

pub trait Seq<T> {
    fn first(&self) -> &T;
    fn rest(&self) -> Option<&Self>;
}

pub fn first<'a, T, S: Seq<T>>(coll: impl Into<Option<&'a S>>) -> Option<&'a T>
where
    S: 'a,
{
    coll.into().and_then(first)
    // match coll.into() {
    //     Some(v) => Some(v.first()),
    //     None => None,
    // }
}

pub fn rest<'a, T, S: Seq<T>>(coll: impl Into<Option<&'a S>>) -> Option<&'a S> {
    coll.into().and_then(rest)
    // match coll.into() {
    //     Some(v) => v.rest(),
    //     None => None,
    // }
}
