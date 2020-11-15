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

type Symbol struct {
	Node
	name string
}

func (s *Symbol) GetType() ast.NodeType {
	return ast.Symbol
}

func (s *Symbol) String() string {
	// TODO: Handle ns qualified symbols here
	return s.name
}

func (s *Symbol) GetName() string {
	// TODO: Handle ns qualified symbols here
	return s.name
}

func (s *Symbol) ToDebugStr() string {
	return s.name
}

func MakeSymbol(s string) *Symbol {
	return &Symbol{
		name: s,
	}
}
