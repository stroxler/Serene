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

// func qqLoop(xs []IExpr) IExpr {
// 	acc := MakeEmptyList()
// 	for i := len(xs) - 1; 0 <= i; i -= 1 {
// 		elem := xs[i]
// 		switch elem.GetType() {
// 		case ast.List:
// 			if ListStartsWith(elem.(*List), "unquote-splicing") {
// 				acc = MakeList([]IExpr{
// 					MakeSymbol(MakeNodeFromExpr(elem), "concat"),
// 					elem.(*List).Rest().First(),
// 					acc})
// 				continue
// 			}
// 		default:
// 		}
// 		acc = MakeList([]IExpr{
// 			MakeSymbol(MakeNodeFromExpr(elem), "cons"),
// 			quasiquote(elem),
// 			acc})
// 	}
// 	return acc
// }

// func quasiquote(e IExpr) IExpr {
// 	switch e.GetType() {
// 	case ast.Symbol:
// 		return MakeList([]IExpr{
// 			MakeSymbol(MakeNodeFromExpr(e), "quote"), e})
// 	case ast.List:
// 		list := e.(*List)
// 		if ListStartsWith(list, "unquote") {
// 			return list.Rest().First()
// 		}
// 		if ListStartsWith(list, "quasiquote") {
// 			return quasiquote(qqLoop(list.ToSlice()))
// 		}

// 		return qqLoop(list.ToSlice())
// 	default:
// 		return e
// 	}
// }
const qqQUOTE string = "*quote*"

func isSymbolEqual(e IExpr, name string) bool {
	if e.GetType() == ast.Symbol && e.(*Symbol).GetName() == name {
		return true
	}
	return false
}
func isQuasiQuote(e IExpr) bool {
	return isSymbolEqual(e, "quasiquote")
}

func isUnquote(e IExpr) bool {
	return isSymbolEqual(e, "unquote")
}

func isUnquoteSplicing(e IExpr) bool {
	return isSymbolEqual(e, "unquote-splicing")
}

func qqSimplify(e IExpr) (IExpr, IError) {
	return e, nil
}

func qqProcess(rt *Runtime, e IExpr) (IExpr, IError) {
	switch e.GetType() {

	// Example: `x => (*quote* x) => (quote x)
	case ast.Symbol:
		sym, err := MakeSymbol(MakeNodeFromExpr(e), qqQUOTE)
		if err != nil {
			//newErr := makeErrorAtPoint()
			// TODO: uncomment next line when we have stackable errors
			// newErr.stack(err)
			return nil, err
		}
		elems := []IExpr{
			sym,
			e,
		}

		return MakeList(
			MakeNodeFromExprs(elems),
			elems,
		), nil

	case ast.List:
		list := e.(*List)
		first := list.First()

		// Example: ``... reads as (quasiquote (quasiquote ...)) and this if will check
		// for the second `quasiquote`
		if isQuasiQuote(first) {
			result, err := qqCompletelyProcess(rt, list.Rest().First())

			if err != nil {
				return nil, err
			}

			return qqProcess(rt, result)
		}

		// Example: `~x reads as (quasiquote (unquote x))
		if isUnquote(first) {
			return list.Rest().First(), nil
		}
		// ???
		if isUnquoteSplicing(first) {
			return nil, MakeError(rt, first, "'unquote-splicing' is not allowed out of a collection.")
		}

		// p := list
		// q := MakeEmptyList()
		// for {
		// 	p = p.Rest().(*List)
		// }

	}

	return e, nil
}

func qqRemoveQQFunctions(e IExpr) (IExpr, IError) {
	return e, nil
}

func qqCompletelyProcess(rt *Runtime, e IExpr) (IExpr, IError) {
	rawResult, err := qqProcess(rt, e)

	if err != nil {
		return nil, err
	}

	if rt.IsQQSimplificationEnabled() {
		rawResult, err = qqSimplify(rawResult)

		if err != nil {
			return nil, err
		}
	}

	return qqRemoveQQFunctions(rawResult)
}

func quasiquote(rt *Runtime, e IExpr) (IExpr, IError) {
	return qqCompletelyProcess(rt, e)
}
