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
	"serene-lang.org/bootstrap/pkg/runtime"
	"serene-lang.org/bootstrap/pkg/scope"
	"serene-lang.org/bootstrap/pkg/types"
)

var sFormsTable = map[string]types.ICallable{
	"def": Def,
}

func GetBuildIn(s *types.Symbol) (types.ICallable, bool) {
	return sFormsTable[s.GetName()]
}

func EvalForm(rt *runtime.Runtime, scope scope.IScope, form types.IExpr) (types.IExpr, error) {
	switch form.GetType() {
	case ast.Nil:
	case ast.Number:
		return form, nil

	// Symbol evaluation rules:
	// * If it's a NS qualified symbol (NSQS), Look it up in the external symbol table of
	// the current namespace.
	// * If it's not a NSQS Look up the name in the current scope.
	// * Otherwise throw an error
	case ast.Symbol:
		symbolName := form.(*types.Symbol).GetName()
		expr := scope.Lookup(symbolName)

		if expr == nil {
			return nil, fmt.Errorf("Can't resolve symbol '%s' in ns '%s'", symbolName, rt.CurrentNS().GetName())
		}

		return expr.Value, nil

	// List evaluation rules:
	// * The first element of the list has to be an expression which implements `ICallable`
	// * The rest o the elements have to be evaluated only after we have determind the the
	//   first element is `ICallable` and it's not a macro or special form.
	// * An empty list evaluates to itself.
	case ast.List:
		list := form.(*types.List)
		if list.Count() == 0 {
			return list, nil
		}
		first := form.(*types.List).First()

		if first.GetType() == ast.Symbol {
			sform, ok := GetBuiltIn(first.(*types.Symbol))
			if ok {
				return sform.Apply(rt, scope, list.Rest())
			}
		}

		fmt.Printf("no builtin %#v", first)

	}

	// Default case
	return nil, errors.New("not implemented")
}

func Eval(rt *runtime.Runtime, forms types.ASTree) (types.IExpr, error) {
	if len(forms) == 0 {
		return &types.Nil, nil
	}

	var ret types.IExpr

	for _, form := range forms {
		// v is here to shut up the linter
		v, err := EvalForm(rt, rt.CurrentNS().GetRootScope(), form)

		if err != nil {
			return nil, err
		}

		ret = v
	}

	return ret, nil
}
