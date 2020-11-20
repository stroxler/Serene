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

// evalForm evaluates the given expression `form` by a slightly different
// evaluation rules. For example if `form` is a list instead of the formal
// evaluation of a list it will evaluate all the elements and return the
// evaluated list
func evalForm(rt *Runtime, scope IScope, form IExpr) (IExpr, error) {

	switch form.GetType() {
	case ast.Nil:
		return form, nil
	case ast.Number:
		return form, nil

	// Symbol evaluation rules:
	// * If it's a NS qualified symbol (NSQS), Look it up in the external symbol table of
	// the current namespace.
	// * If it's not a NSQS Look up the name in the current scope.
	// * Otherwise throw an error
	case ast.Symbol:
		symbolName := form.(*Symbol).GetName()
		expr := scope.Lookup(symbolName)

		if expr == nil {
			return nil, fmt.Errorf("can't resolve symbol '%s' in ns '%s'", symbolName, rt.CurrentNS().GetName())
		}

		return expr.Value, nil

	// Evaluate all the elements in the list instead of following the lisp convention
	case ast.List:
		var result []IExpr

		lst := form.(*List)

		for {
			if lst.Count() > 0 {
				expr, err := EvalForms(rt, scope, lst.First())
				if err != nil {
					return nil, err
				}
				result = append(result, expr)
				lst = lst.Rest().(*List)
			} else {
				break
			}
		}
		return MakeList(result), nil
	}

	// Default case
	return nil, errors.New("not implemented")

}

// EvalForms evaluates the given expr `expressions` (it can be a list, block, symbol or anything else)
// with the given runtime `rt` and the scope `scope`.
func EvalForms(rt *Runtime, scope IScope, expressions IExpr) (IExpr, error) {
	// EvalForms is the main and the most important evaluation function on Serene.
	// It's a long loooooooooooong function. Why? Well, Because we don't want to
	// waste call stack spots in order to have a well organized code.
	// In order to avoid stackoverflows and implement TCO ( which is a must for
	// a functional language we need to avoid unnecessary calls and keep as much
	// as possible in a loop.
	var ret IExpr
	var err error

tco:
	for {
		// The TCO loop is there to take advantage or the fact that
		// in order to call a function or a block we simply can change
		// the value of the `expressions` and `scope`
		var exprs []IExpr

		// Block evaluation rules:
		// * If empty, return Nothing
		// * Otherwise evaluate the expressions in the block one by one
		//   and return the last result
		if expressions.GetType() == ast.Block {
			if expressions.(*Block).Count() == 0 {
				return &Nothing, nil
			}
			exprs = expressions.(*Block).ToSlice()
		} else {
			exprs = []IExpr{expressions}
		}

	body:
		for _, forms := range exprs {
			// Evaluating forms one by one

			if forms.GetType() != ast.List {
				ret, err = evalForm(rt, scope, forms)
				break tco // return ret, err
			}

			list := forms.(*List)

			// Empty list evaluates to itself
			if list.Count() == 0 {
				ret = list
				break tco // return &Nil, nil
			}

			rawFirst := list.First()
			sform := ""

			// Handling special forms by looking up the first
			// element of the list. If it is a symbol, Grab
			// the name and check it for build it forms.
			//
			// Note: If we don't care about recursion in any
			// case we can simply extract it to a function
			// for example in `def` since we are going to
			// evaluate the value separately, we don't care
			// about recursion because we're going to handle
			// it wen we're evaluating the value. But in the
			// case of let it's a different story.
			if rawFirst.GetType() == ast.Symbol {
				sform = rawFirst.(*Symbol).GetName()
			}

			switch sform {
			case "def":
				ret, err = Def(rt, scope, list.Rest().(*List))
				break tco // return

			case "fn":
				ret, err = Fn(rt, scope, list.Rest().(*List))
				break tco // return

			// List evaluation rules:
			// * The first element of the list has to be an expression which is callable
			// * An empty list evaluates to itself.
			default:
				// Evaluating all the elements of the list
				exprs, e := evalForm(rt, scope, list)
				if e != nil {
					err = e
					ret = nil
					break tco //return
				}

				f := exprs.(*List).First()

				switch f.GetType() {
				case ast.Fn:
					// If the first element of the evaluated list is a function
					// create a scope for it by creating the binding to the given
					// parameters in the new scope and set the parent of it to
					// the scope which the function defined in and then set the
					// `expressions` to the body of function and loop again
					fn := f.(*Function)
					if e != nil {
						err = e
						ret = nil
						break body //return

					}

					argList := exprs.(*List).Rest().(*List)

					scope, e = MakeFnScope(fn.GetScope(), fn.GetParams(), argList)
					if e != nil {
						err = e
						ret = nil
						break body //return
					}

					expressions = fn.GetBody()
					continue tco
				default:
					err = errors.New("don't know how to execute anything beside function")
					ret = nil
					break tco
				}
			}
		}
	}

	return ret, err
}

// Eval the given `Block` of code with the given runtime `rt`.
// The Important part here is that any expression that we need
// to Eval has to be wrapped in a Block. Don't confused the
// concept of Block with blocks from other languages which
// specify by using `{}` or indent or what ever. Blocks in terms
// of Serene are just arrays of expressions and nothing more.
func Eval(rt *Runtime, forms *Block) (IExpr, error) {
	if forms.Count() == 0 {
		// Nothing is literally Nothing
		return &Nothing, nil
	}

	v, err := EvalForms(rt, rt.CurrentNS().GetRootScope(), forms)

	if err != nil {
		return nil, err
	}

	return v, nil
}
