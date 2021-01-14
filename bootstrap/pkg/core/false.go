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

import (
	"serene-lang.org/bootstrap/pkg/ast"
	"serene-lang.org/bootstrap/pkg/hash"
)

type False struct {
	Node
	ExecutionScope
}

func (f *False) GetType() ast.NodeType {
	return ast.False
}

func (f *False) String() string {
	return "false"
}

func (f *False) ToDebugStr() string {
	return "false"
}

func (f *False) Hash() uint32 {
	bytes := []byte("false")
	return hash.Of(append([]byte{byte(ast.False)}, bytes...))
}

func MakeFalse(n Node) *False {
	return &False{Node: n}
}
