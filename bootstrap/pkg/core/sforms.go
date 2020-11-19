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

	"serene-lang.org/bootstrap/pkg/ast"
)

func Fn(rt *Runtime, scope IScope, args *List) (IExpr, error) {

	if args.Count() < 1 {
		return nil, errors.New("'fn' needs at least an arguments list")
	}

	var params IColl
	body := MakeEmptyBlock()

	arguments := args.First()

	// TODO: Add vector in here
	// Or any other icoll
	if arguments.GetType() == ast.List {
		params = arguments.(IColl)
	}

	if args.Count() > 1 {
		body.SetContent(args.Rest().(*List).ToSlice())
	}

	return MakeFunction(scope, params, body), nil

}
