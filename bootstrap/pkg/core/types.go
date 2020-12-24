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
	"fmt"

	"serene-lang.org/bootstrap/pkg/ast"
	"serene-lang.org/bootstrap/pkg/hash"
)

// IRepresentable is the interface which any value that wants to have a string
// representation has to implement. Serene will use this string where ever
// it needs to present a value as string.
type IRepresentable interface {
	fmt.Stringer
}

// IPrintable is the interface which any value that wants to have a string
// representation for printing has to implement. The `print` family functions will use
// this interface to convert forms to string first and if the value doesn't
// implement this interface they will resort to `IRepresentable`
type IPrintable interface {
	IRepresentable
	PrintToString() string
}

// IDebuggable is the interface designed for converting forms to a string
// form which are meant to be used as debug data
type IDebuggable interface {
	ToDebugStr() string
}

// IExpr is the most important interface in Serene which basically represents
// a VALUE in Serene. All the forms (beside special formss) has to implement
// this interface.
type IExpr interface {
	ast.ILocatable
	ast.ITypable
	hash.IHashable
	IRepresentable
	IDebuggable
}

// Node struct is simply representing a Node in the AST which provides the
// functionalities required to trace the code based on the location.
type Node struct {
	location ast.Location
}

// GetLocation returns the location of the Node in the source input
func (n Node) GetLocation() ast.Location {
	return n.location
}

// Helper functions ===========================================================

// toRepresentables converts the given collection of IExprs to an array of
// IRepresentable. Since golangs type system is weird ( if A is an interface
// that embeds interface B you []A should be usable as []B but that's not the
// case in Golang), we need this convertor helper
func toRepresentables(ast IColl) []IRepresentable {
	var params []IRepresentable

	for _, x := range ast.ToSlice() {
		params = append(params, x.(IRepresentable))
	}

	return params
}

// MakeNodeFromLocation creates a new Node for the given Location `loc`
func MakeNodeFromLocation(loc ast.Location) Node {
	return Node{
		location: loc,
	}
}

// MakeNodeFromExpr creates a new Node from the given `IExpr`.
// We use the Node to pass it to other IExpr constructors to
// keep the reference to the original form in the input string
func MakeNodeFromExpr(e IExpr) Node {
	return MakeNodeFromLocation(e.GetLocation())
}

// MakeNode creates a new Node in the the given `input` that points to a
// range of characters starting from the `start` till the `end`.
func MakeNode(input *[]string, start int, end int) Node {
	return MakeNodeFromLocation(ast.MakeLocation(input, start, end))
}

// MakeSinglePointNode creates a not the points to a single char in the
// input
func MakeSinglePointNode(input *[]string, point int) Node {
	return MakeNode(input, point, point)
}
