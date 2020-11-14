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
	"fmt"

	"serene-lang.org/bootstrap/pkg/types"
)

type IScope interface {
	Lookup(k string) (*Binding, error)
	Insert(k string, v types.IExpr, public bool)
}

type Binding struct {
	value  types.IExpr
	public bool
}

type Scope struct {
	bindings map[string]*Binding
	parent   IScope
}

func (s *Scope) Lookup(k string) (*Binding, error) {
	v, ok := s.bindings[k]
	if ok {
		return v, nil
	}

	if s.parent != nil {
		return s.parent.Lookup(k)
	} else {
		return nil, fmt.Errorf("can't resolve symbol '%s'", k)
	}
}

func (s *Scope) Insert(k string, v types.IExpr, public bool) {
	s.bindings[k] = &Binding{value: v, public: public}
}

func MakeScope(parent *Scope) Scope {
	return Scope{
		parent:   parent,
		bindings: map[string]*Binding{},
	}
}
