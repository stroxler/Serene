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

// Package eval provides all the necessary functions to eval expressions
package eval

import (
	"serene-lang.org/bootstrap/pkg/runtime"
	"serene-lang.org/bootstrap/pkg/types"
)

func eval(rt *runtime.Runtime, forms types.ASTree) types.IExpr {
	if len(forms) == 0 {
		return &types.Nil
	}

	var ret types.IExpr

	for _, form := range forms {
		ret = eval_form(rt, rt.CurrentNS().GetRootScope(), form)
	}

	ret
}
