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
	"strings"
)

// BUILTINS is used in the Runtime to support builtin functions of
// the language which are implemented in Go
var BUILTINS = map[string]NativeFunction{
	"print":   MakeNativeFn("print", PrintNativeFn),
	"require": MakeNativeFn("print", RequireNativeFn),
	"hash":    MakeNativeFn("hash", HashNativeFn),
}

func PrintNativeFn(rt *Runtime, scope IScope, n Node, args *List) (IExpr, IError) {
	var params []string

	for _, expr := range args.Rest().(*List).ToSlice() {
		params = append(params, expr.String())
	}

	fmt.Print(strings.Join(params, " "))
	return &Nil, nil
}

func RequireNativeFn(rt *Runtime, scope IScope, n Node, args *List) (IExpr, IError) {
	switch args.Count() {
	case 0:
		return nil, MakeErrorFor(rt, args, "'require' function is missing")
	case 1:
		return nil, MakeErrorFor(rt, args.First(), "'require' function needs at least one argument")
	default:
	}

	var result IExpr
	var err IError
	for _, ns := range args.Rest().(*List).ToSlice() {
		result, err = RequireNamespace(rt, ns)
		if err != nil {
			return nil, err
		}
	}

	return result, nil

}

func HashNativeFn(rt *Runtime, scope IScope, n Node, args *List) (IExpr, IError) {
	if args.Count() != 2 {
		return nil, MakeErrorFor(rt, args.First(), "'hash' function needs exactly one argument")
	}

	expr := args.Rest().First()
	result, err := MakeInteger(expr.Hash())

	if err != nil {
		return nil, err
	}

	result.Node = n
	return result, nil
}
