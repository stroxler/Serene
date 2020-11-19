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

type INamespace interface {
	DefineGlobal()
	LookupGlobal()
	GetRootScope() IScope
	// return the fully qualified name of the namespace
	GetName() string
}

type Namespace struct {
	name      string
	rootScope Scope
	source    string
	externals map[string]Namespace
}

func (n *Namespace) DefineGlobal(k string, v IExpr, public bool) {
	n.rootScope.Insert(k, v, public)
}

func (n *Namespace) LookupGlobal() {}

func (n *Namespace) GetRootScope() IScope {
	return &n.rootScope
}

func (n *Namespace) GetName() string {
	return n.name
}

func MakeNS(name string, source string) Namespace {
	s := MakeScope(nil)
	return Namespace{
		name:      name,
		rootScope: *s,
		source:    source,
		externals: map[string]Namespace{},
	}
}
