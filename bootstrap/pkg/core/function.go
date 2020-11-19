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

type ICallable interface {
	Apply(rt *Runtime, scope IScope, args *List) (IExpr, error)
}

type Function struct {
	Node
	name   string
	scope  IScope
	params IColl
	body   *Block
}

func (f *Function) GetType() ast.NodeType {
	return ast.Fn
}

func (f *Function) String() string {
	return fmt.Sprintf("<Fn: %s at %p", f.name, f)
}

func (f *Function) GetName() string {
	// TODO: Handle ns qualified symbols here
	return f.name
}

func (f *Function) GetScope() IScope {
	return f.scope
}

func (f *Function) GetParams() IColl {
	return f.params
}

func (f *Function) ToDebugStr() string {
	return fmt.Sprintf("<Fn: %s at %p", f.name, f)
}

func (f *Function) GetBody() *Block {
	return f.body
}

func MakeFunction(scope IScope, params IColl, body *Block) *Function {
	return &Function{
		scope:  scope,
		params: params,
		body:   body,
	}
}

func MakeFnScope(parent IScope, bindings IColl, values IColl) (*Scope, error) {
	fmt.Printf("%s    %s\n", bindings, values)
	scope := MakeScope(parent.(*Scope))

	// TODO: Implement destructuring
	if bindings.Count() > values.Count() {
		return nil, errors.New("'binding' and 'valuse' size don't match")
	}

	binds := bindings.ToSlice()
	exprs := values.ToSlice()

	for i := 0; i < len(binds); i += 1 {
		if binds[i].GetType() == ast.Symbol && binds[i].(*Symbol).IsRestable() {
			scope.Insert(binds[i+1].(*Symbol).GetName(), MakeList(exprs[i:]), false)
			break
		} else {
			scope.Insert(binds[i].(*Symbol).GetName(), exprs[i], false)
		}
	}

	return scope, nil
}
