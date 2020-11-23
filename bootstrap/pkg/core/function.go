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
	"errors"
	"fmt"

	"serene-lang.org/bootstrap/pkg/ast"
)

// Function struct represent a user defined function.
type Function struct {
	// Node struct holds the necessary functions to make
	// Functions locatable
	Node

	// Name of the function, it can be empty and it has to be
	// set via `def`
	name string

	// Parent scope of the function. The scope which the function
	// is defined in
	scope IScope

	// A collection of arguments. Why IColl? because we can use
	// Lists and Vectors for the argument lists. Maybe even
	// hashmaps in future.
	params IColl

	// A reference to the body block of the function
	body *Block
}

func (f *Function) GetType() ast.NodeType {
	return ast.Fn
}

func (f *Function) String() string {
	return fmt.Sprintf("<Fn: %s at %p", f.name, f)
}

func (f *Function) GetName() string {
	// TODO: Handle ns qualified symbols here
	return f.name
}

func (f *Function) GetScope() IScope {
	return f.scope
}

func (f *Function) GetParams() IColl {
	return f.params
}

func (f *Function) ToDebugStr() string {
	return fmt.Sprintf("<Fn: %s at %p", f.name, f)
}

func (f *Function) GetBody() *Block {
	return f.body
}

// MakeFunction Create a function with the given `params` and `body` in
// the given `scope`.
func MakeFunction(scope IScope, params IColl, body *Block) *Function {
	return &Function{
		scope:  scope,
		params: params,
		body:   body,
	}
}

// MakeFnScope a new scope for the body of a function. It binds the `bindings`
// to the given `values`.
func MakeFnScope(parent IScope, bindings IColl, values IColl) (*Scope, error) {
	scope := MakeScope(parent.(*Scope))

	// TODO: Implement destructuring

	if bindings.Count() > values.Count() {
		return nil, errors.New("'binding' and 'valuse' size don't match")
	}

	binds := bindings.ToSlice()
	exprs := values.ToSlice()

	for i := 0; i < len(binds); i += 1 {
		// If an argument started with char `&` use it to represent
		// rest of values.
		//
		// for example: `(fn (x y &z) ...)`
		if binds[i].GetType() == ast.Symbol && binds[i].(*Symbol).IsRestable() {
			scope.Insert(binds[i+1].(*Symbol).GetName(), MakeList(exprs[i:]), false)
			break
		} else {
			scope.Insert(binds[i].(*Symbol).GetName(), exprs[i], false)
		}
	}

	return scope, nil
}
