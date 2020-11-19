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

func EvalForms(rt *Runtime, scope IScope, forms IExpr) (IExpr, error) {
	for {
		if forms.GetType() != ast.List {
			return EvalForm(rt, scope, forms)
		}

		list := forms.(*List)

		if list.Count() == 0 {
			return &Nil, nil
		}
		fmt.Printf("EVAL: %s\n", list)

		rawFirst := list.First()
		sform := ""

		// Handling special forms
		if rawFirst.GetType() == ast.Symbol {
			sform = rawFirst.(*Symbol).GetName()
		}

		fmt.Printf("sform: %s\n", sform)
		switch sform {

		case "def":
			return Def(rt, scope, list.Rest().(*List))

		case "fn":
			return Fn(rt, scope, list.Rest().(*List))

		default:
			exprs, err := EvalForm(rt, scope, list)
			if err != nil {
				return nil, err
			}
			f := exprs.(*List).First()

			switch f.GetType() {
			case ast.Fn:
				fn := f.(*Function)
				// Since we're passing a List to evaluate the
				// result has to be a list. ( Rest returns a List )
				args, e := EvalForm(rt, scope, list.Rest().(*List))
				if e != nil {
					return nil, e
				}
				argList, _ := args.(*List)

				scope, err = MakeFnScope(fn.GetScope(), fn.GetParams(), argList)
				if err != nil {
					return nil, err
				}
				forms = MakeList(fn.GetBody().ToSlice())
			// case ast.InteropFn:
			default:
				lst := exprs.(*List)
				return lst.ToSlice()[lst.Count()-1], nil
				// TODO: Fix this ugly error msg
				//return nil, fmt.Errorf("can't call anything beside functions yet")

			}
		}
	}

}

func Eval(rt *Runtime, forms ASTree) (IExpr, error) {
	if len(forms) == 0 {
		return &Nil, nil
	}

	v, err := EvalForm(rt, rt.CurrentNS().GetRootScope(), MakeList(forms))

	if err != nil {
		return nil, err
	}

	return v.(*List).First(), nil
}