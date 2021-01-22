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

type Bool struct {
	Node
	ExecutionScope
	value bool
}

func (t *Bool) GetType() ast.NodeType {
	return ast.Bool
}

func (t *Bool) String() string {
	if t.value {
		return TRUEFORM
	}
	return FALSEFORM
}

func (t *Bool) ToDebugStr() string {
	return t.String()
}

func (t *Bool) Hash() uint32 {
	bytes := []byte(t.String())
	return hash.Of(append([]byte{byte(ast.Bool)}, bytes...))
}

func (t *Bool) IsTrue() bool {
	return t.value
}

func (t *Bool) IsFalse() bool {
	return !t.value
}

func MakeTrue(n Node) *Bool {
	return &Bool{Node: n, value: true}
}

func MakeFalse(n Node) *Bool {
	return &Bool{Node: n, value: false}
}
