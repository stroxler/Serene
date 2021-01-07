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

import "fmt"

type IScope interface {
	Lookup(rt *Runtime, k string) *Binding
	Insert(k string, v IExpr, public bool)
	GetNS(rt *Runtime) *Namespace
	SetNS(ns *string)
}

type Binding struct {
	Value  IExpr
	Public bool
}

type Scope struct {
	bindings map[string]Binding
	parent   *Scope
	ns       *string
}

func (s *Scope) Lookup(rt *Runtime, k string) *Binding {
	if rt.IsDebugMode() {
		fmt.Printf("[DEBUG] Looking up '%s'\n", k)
	}

	v, ok := s.bindings[k]
	if ok {
		if rt.IsDebugMode() {
			fmt.Printf("[DEBUG] Found '%s': '%s'\n", k, v.Value.String())
		}
		return &v
	}

	if s.parent != nil {
		return s.parent.Lookup(rt, k)
	}

	builtin := rt.LookupBuiltin(k)

	if builtin != nil {
		if rt.IsDebugMode() {
			fmt.Printf("[DEBUG] Found builtin '%s': '%s'\n", k, builtin)
		}

		return &Binding{builtin, true}
	}

	return nil
}

func (s *Scope) Insert(k string, v IExpr, public bool) {
	s.bindings[k] = Binding{Value: v, Public: public}
}

func (s *Scope) GetNS(rt *Runtime) *Namespace {
	if s.ns == nil {
		panic("A scope with no namespace !!!!")
	}
	ns, ok := rt.GetNS(*s.ns)
	if !ok {
		panic(fmt.Sprintf("A scope with the wrong namespace! '%s'", s.ns))
	}
	return ns
}

func (s *Scope) SetNS(ns *string) {
	s.ns = ns
}

func MakeScope(rt *Runtime, parent *Scope, namespace *string) *Scope {
	var belongsTo *string

	if parent != nil {
		nsName := parent.GetNS(rt).GetName()
		belongsTo = &nsName
	} else if namespace == nil {
		panic("When the 'parent' is nil, you have to provide the 'namespace' name.")
	} else {
		belongsTo = namespace
	}

	return &Scope{
		parent:   parent,
		bindings: map[string]Binding{},
		ns:       belongsTo,
	}
}
