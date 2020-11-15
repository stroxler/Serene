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
	"serene-lang.org/bootstrap/pkg/runtime"
	"serene-lang.org/bootstrap/pkg/scope"
	"serene-lang.org/bootstrap/pkg/types"
)

/** TODO:
Make sure to implement INode for Def as well to be able to point
to the write place at the input stream for error messages.
*/

type def struct{}

var Def = def{}

func (d def) Apply(rt *runtime.Runtime, scope scope.IScope, args *types.List) (types.IExpr, error) {
	switch args.Count() {
	case 2:
		name := args.First()

		if name.GetType() != ast.Symbol {
			return nil, errors.New("The first argument of 'def' has to be a symbol")
		}

		sym := name.(*types.Symbol)

		//value = args.Rest().(*types.List).First()
		valueExpr := args.Rest().First()
		value, err := EvalForm(rt, scope, valueExpr)

		if err != nil {
			return nil, err
		}

		ns := rt.CurrentNS()
		ns.DefineGlobal(sym.GetName(), value, true)
		return sym, nil
	}

	return nil, errors.New("'def' form need at least 2 arguments")
}
