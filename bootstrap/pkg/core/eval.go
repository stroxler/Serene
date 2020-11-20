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

func EvalForm(rt *Runtime, scope IScope, form IExpr) (IExpr, error) {

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

	// List evaluation rules:
	// * The first element of the list has to be an expression which implements `ICallable`
	// * The rest o the elements have to be evaluated only after we have determind the the
	//   first element is `ICallable` and it's not a macro or special form.
	// * An empty list evaluates to itself.
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

func EvalForms(rt *Runtime, scope IScope, expressions IExpr) (IExpr, error) {
	var ret IExpr
	var err error

tco:
	for {
		var exprs []IExpr

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

			switch forms.GetType() {
			case ast.List:

			default:
				ret, err = EvalForm(rt, scope, forms)
				break tco // return
			}

			if forms.(*List).Count() == 0 {
				ret = &Nil
				break tco // return
			}

			list := forms.(*List)
			rawFirst := list.First()
			sform := ""

			// Handling special forms
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
			default:
				exprs, e := EvalForm(rt, scope, list)
				if e != nil {
					err = e
					ret = nil
					break tco //return
				}

				f := exprs.(*List).First()

				switch f.GetType() {
				case ast.Fn:
					fn := f.(*Function)
					if e != nil {
						err = e
						ret = nil
						break body //return

					}
					//argList, _ := args.(*List)
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

func Eval(rt *Runtime, forms *Block) (IExpr, error) {
	if forms.Count() == 0 {
		return &Nothing, nil
	}

	v, err := EvalForms(rt, rt.CurrentNS().GetRootScope(), forms)

	if err != nil {
		return nil, err
	}

	return v, nil
}
