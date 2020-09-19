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

/// Seq trait describes a seqable collection. Please note that `first`
/// and `rest` return a copy of the data not the reference!
pub trait Seq<T> {
    type Coll;

    fn first(&self) -> Option<T>;
    fn rest(&self) -> Self::Coll;
}

// pub fn first<'a, T, S: Seq<T>>(coll: impl Into<Option<&'a S>>) -> Option<&'a T>
// where
//     S: 'a,
// {
//     coll.into().and_then(first)
// }

// pub fn rest<'a, T, S: Seq<T>>(coll: impl Into<Option<&'a S>>) -> S {
//     match coll.into() {
//         Some(e) => e.rest(),
//         None =>
//     }
// }
