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
use crate::ast::Node;
use std::fmt;

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,

    /// This field holds the ns specifier part of the symbol. For example
    /// in case of `somens/somesym`, this field will hold `somens`.
    pub target_ns: Option<String>,
}

impl Symbol {
    pub fn new(name: String, target_ns: Option<String>) -> Symbol {
        Symbol { name, target_ns }
    }

    pub fn is_ns_qualified(&self) -> bool {
        !self.target_ns.is_none()
    }

    pub fn is_def(&self) -> bool {
        self.name == "def"
    }
}

impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Symbol {}

impl Node for Symbol {
    fn get_type_str(&self) -> &str {
        "Symbol"
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.name)
    }
}
