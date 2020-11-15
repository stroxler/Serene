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

// Package scope provides several interfaces and their implementations
// such `Scope` which acts as the environment in the Lisp literature.
package scope

import (
	"serene-lang.org/bootstrap/pkg/types"
)

type IScope interface {
	Lookup(k string) *Binding
	Insert(k string, v types.IExpr, public bool)
}

type Binding struct {
	Value  types.IExpr
	Public bool
}

type Scope struct {
	bindings map[string]Binding
	parent   *Scope
}

func (s *Scope) Lookup(k string) *Binding {
	v, ok := s.bindings[k]
	if ok {
		return &v
	}

	if s.parent != nil {
		return s.parent.Lookup(k)
	}

	return nil
}

func (s *Scope) Insert(k string, v types.IExpr, public bool) {
	s.bindings[k] = Binding{Value: v, Public: public}
}

func MakeScope(parent *Scope) Scope {
	return Scope{
		parent:   parent,
		bindings: map[string]Binding{},
	}
}
