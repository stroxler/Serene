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

// Function implementations:
// * We have two different types of functions. User defined functions and
//   native functions
// * User defined functions are represented via the `Function` struct.
// * Native functions are represented via the `NativeFunction` struct.
// * User defined functions gets evaluated by:
// - Creating a new scope a direct child of the scope that the function
//   defined in
// - Creating bindings in the new scope to bind the passed values to their
//   arguments names
// - Evaluate the body of the function in context of the new scope and return
//   the result of the last expression
// * Native functions evaluates by calling the `Apply` method of the `IFn`
//   interface which is quite simple.
//
// TODOs:
// * Support for multi-arity functions
// * Support for protocol functions
// * `IFn` protocol

import (
	"fmt"

	"serene-lang.org/bootstrap/pkg/ast"
	"serene-lang.org/bootstrap/pkg/errors"
	"serene-lang.org/bootstrap/pkg/hash"
)

type nativeFnHandler = func(rt *Runtime, scope IScope, n Node, args *List) (IExpr, IError)

type IFn interface {
	IExpr
	Apply(rt *Runtime, scope IScope, n Node, args *List) (IExpr, IError)
	GetName() string
}

// Function struct represent a user defined function.
type Function struct {
	// Node struct holds the necessary functions to make
	// Functions locatable
	Node
	ExecutionScope
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
	body    *Block
	isMacro bool
}

type NativeFunction struct {
	// Node struct holds the necessary functions to make
	// Functions locatable
	Node
	ExecutionScope
	name string
	fn   nativeFnHandler
}

func (f *Function) GetType() ast.NodeType {
	return ast.Fn
}

func (f *Function) Hash() uint32 {
	// TODO: Fix this function to return an appropriate hash for a function
	return hash.Of([]byte(f.String()))
}

func (f *Function) IsMacro() bool {
	return f.isMacro
}

func (f *Function) String() string {
	if f.isMacro {
		return fmt.Sprintf("<Macro: %s at %p>", f.name, f)
	}

	return fmt.Sprintf("<Fn: %s at %p>", f.name, f)
}

func (f *Function) GetName() string {
	// TODO: Handle ns qualified symbols here
	return f.name
}

func (f *Function) SetName(name string) {
	f.name = name
}

func (f *Function) GetScope() IScope {
	return f.scope
}

func (f *Function) GetParams() IColl {
	return f.params
}

func (f *Function) ToDebugStr() string {
	if f.isMacro {
		return fmt.Sprintf("<Macro: %s at %p>", f.name, f)
	}

	return fmt.Sprintf("<Fn: %s at %p>", f.name, f)
}

func (f *Function) GetBody() *Block {
	return f.body
}

func (f *Function) Apply(rt *Runtime, scope IScope, n Node, args *List) (IExpr, IError) {
	application := args.Cons(f)
	return EvalForms(rt, scope, application)
}

// MakeFunction Create a function with the given `params` and `body` in
// the given `scope`.
func MakeFunction(n Node, scope IScope, params IColl, body *Block) *Function {
	return &Function{
		Node:    n,
		scope:   scope,
		params:  params,
		body:    body,
		isMacro: false,
	}
}

// MakeFnScope a new scope for the body of a function. It binds the `bindings`
// to the given `values`.
func MakeFnScope(rt *Runtime, parent IScope, bindings, values IColl) (*Scope, IError) { //nolint:gocyclo
	// TODO: Break this function into smaller functions
	scope := MakeScope(rt, parent.(*Scope), nil)
	// TODO: Implement destructuring
	binds := bindings.ToSlice()
	exprs := values.ToSlice()
	numberOfBindings := len(binds)

	if len(binds) > 0 {
		lastBinding := binds[len(binds)-1]

		if lastBinding.GetType() == ast.Symbol && lastBinding.(*Symbol).IsRestable() {
			numberOfBindings = len(binds) - 1
		}

		if lastBinding.GetType() == ast.Symbol && !lastBinding.(*Symbol).IsRestable() && numberOfBindings < len(exprs) {
			return nil, MakeSemanticError(
				rt,
				values.(IExpr),
				errors.E0002,
				fmt.Sprintf("expected '%d' arguments, got '%d'.", bindings.Count(), values.Count()),
			)
		}
	}

	if numberOfBindings > len(exprs) {
		if rt.IsDebugMode() {
			fmt.Printf("[DEBUG] Mismatch on bindings and values: Bindings: %s, Values: %s\n", bindings, values)
		}

		return nil, MakeSemanticError(
			rt,
			values.(IExpr),
			errors.E0002,
			fmt.Sprintf("expected '%d' arguments, got '%d'.", bindings.Count(), values.Count()),
		)
	}

	for i := 0; i < len(binds); i++ {
		// If an argument started with char `&` use it to represent
		// rest of values.
		//
		// for example: `(fn (x y &z) ...)`
		if binds[i].GetType() == ast.Symbol && binds[i].(*Symbol).IsRestable() {
			if i != len(binds)-1 {
				return nil, MakeError(rt, binds[i], "The function argument with '&' has to be the last argument.")
			}

			// if the number of values are one less than the number of bindings
			// but the last binding is a Restable (e.g &x) the the last bindings
			// has to be an empty list. Note the check for number of vlaues comes
			// next.
			rest := MakeEmptyList(MakeNodeFromExpr(binds[i]))

			if i <= len(exprs)-1 {
				// If the number of values matches the number of bindings
				// or it is more than that create a list from them
				// to pass it to the last argument that has to be Restable (e.g &x)
				elements := exprs[i:]
				var node Node

				if len(elements) > 0 {
					n := MakeNodeFromExprs(elements)

					if n == nil {
						n = &values.(*List).Node
					}

					node = *n
				} else {
					node = MakeNodeFromExpr(binds[i])
				}

				rest = MakeList(node, elements)
			}

			scope.Insert(binds[i].(*Symbol).GetName()[1:], rest, false)
			break
		} else {
			scope.Insert(binds[i].(*Symbol).GetName(), exprs[i], false)
		}
	}

	return scope, nil
}

func (f *NativeFunction) GetType() ast.NodeType {
	return ast.NativeFn
}

func (f *NativeFunction) GetName() string {
	return f.name
}

func (f *NativeFunction) String() string {
	return fmt.Sprintf("<NativeFn: %s at %p>", f.name, f)
}

func (f *NativeFunction) Hash() uint32 {
	// TODO: Fix this function to return an appropriate hash for a function
	return hash.Of([]byte(f.String()))
}

func (f *NativeFunction) ToDebugStr() string {
	return fmt.Sprintf("<NativeFn: %s>", f.name)
}

func (f *NativeFunction) Apply(rt *Runtime, scope IScope, n Node, args *List) (IExpr, IError) {
	return f.fn(rt, scope, n, args)
}

func MakeNativeFn(name string, f nativeFnHandler) NativeFunction {
	return NativeFunction{
		Node: MakeNodeFromLocation(ast.MakeUnknownLocation()),
		name: name,
		fn:   f,
	}
}
