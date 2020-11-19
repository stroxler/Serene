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

// Package types provides the type interface of Serene. All the types
// in Serene are directly AST Nodes as well.
package core

import (
	"fmt"

	"serene-lang.org/bootstrap/pkg/ast"
)

type IPrintable interface {
	fmt.Stringer
}

type IDebuggable interface {
	ToDebugStr() string
}

type IExpr interface {
	ast.ILocatable
	ast.ITypable
	IPrintable
	IDebuggable
}

type Node struct {
	location int
}

func (n Node) GetLocation() int {
	return n.location
}
