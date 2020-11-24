/*
 Serene --- Yet an other Lisp

Copyright (c) 2020  Sameer Rahmani <lxsameer@gnu.org>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

package core

import "serene-lang.org/bootstrap/pkg/ast"

type NothingType struct{}

var Nothing = NothingType{}

func (n NothingType) GetType() ast.NodeType {
	return ast.Nothing
}

func (n NothingType) GetLocation() ast.Location {
	return ast.MakeUnknownLocation()
}

func (n NothingType) String() string {
	return ""
}

func (n NothingType) ToDebugStr() string {
	return ""
}