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

// TODO:
// * Add support for `before` and `after` state in macroexpantion
//   and call stack. So in case of an error. Users should be able
//   to see the forms before and after expansion.

import (
	"serene-lang.org/bootstrap/pkg/ast"
)

// Serene macros are in fact functions with the `isMacro` flag set to true.
// We have only normal macro implementation in bootstrap version of serene in
// compare to reader macros and evaluator macros and and other types.

// MakeMacro creates a macro with the given `params` and `body` in
// the given `scope`.
func MakeMacro(scope IScope, name string, params IColl, body *Block) *Function {
	return &Function{
		name:    name,
		scope:   scope,
		params:  params,
		body:    body,
		isMacro: true,
	}
}

// isMacroCall looks up the given `form` in the given `scope` if it is a symbol.
// If there is a value associated with the symbol in the scope, it will be checked
// to be a macro.
func isMacroCall(rt *Runtime, scope IScope, form IExpr) (*Function, bool) {
	if form.GetType() == ast.List {
		list := form.(*List)
		if list.Count() == 0 {
			return nil, false
		}

		first := list.First()
		var macro IExpr = nil

		if first.GetType() == ast.Symbol {
			binding := scope.Lookup(rt, first.(*Symbol).GetName())
			if binding != nil && binding.Public {
				macro = binding.Value
			}
		}
		if macro != nil {
			if macro.GetType() == ast.Fn && macro.(*Function).IsMacro() {
				return macro.(*Function), true
			}
		}
	}
	return nil, false
}

// applyMacro works very similar to how we evaluate function calls the only difference
// is that we don't evaluate the arguments and create the bindings in the scope of the
// body directly as they are. It's Lisp Macroes after all.
func applyMacro(rt *Runtime, macro *Function, args *List) (IExpr, IError) {
	mscope, e := MakeFnScope(rt, macro.GetScope(), macro.GetParams(), args)

	if e != nil {
		return nil, e
	}

	return EvalForms(rt, mscope, macro.GetBody())
}

// macroexpand expands the given `form` as a macro and returns the resulted
// expression
func macroexpand(rt *Runtime, scope IScope, form IExpr) (IExpr, IError) {
	var macro *Function
	var e IError
	ok := false
	//form := expr

	for {
		macro, ok = isMacroCall(rt, scope, form)
		if !ok {
			return form, nil
		}

		form, e = applyMacro(rt, macro, form.(IColl).Rest().(*List))

		if e != nil {
			return nil, e
		}
	}
}
