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

	"serene-lang.org/bootstrap/pkg/ast"
)

type INamespace interface {
	DefineGlobal()
	LookupGlobal()
	GetRootScope() IScope
	// return the fully qualified name of the namespace
	GetName() string
	getForms() *Block
	setForms(forms *Block)
}

type Namespace struct {
	name      string
	rootScope Scope
	source    string
	externals map[string]*Namespace
	forms     Block
}

func (n *Namespace) GetType() ast.NodeType {
	return ast.Namespace
}

func (n *Namespace) GetLocation() ast.Location {
	return ast.MakeUnknownLocation()
}

func (n *Namespace) String() string {
	return fmt.Sprintf("<ns: %s at %s>", n.name, n.source)
}

func (n *Namespace) ToDebugStr() string {
	return fmt.Sprintf("<ns: %s at %s>", n.name, n.source)
}

func (n *Namespace) DefineGlobal(k string, v IExpr, public bool) {
	n.rootScope.Insert(k, v, public)
}

func (n *Namespace) LookupGlobal(sym *Symbol) *Binding {
	if !sym.IsNSQualified() {
		return nil
	}

	externalNS, ok := n.externals[sym.GetNSPart()]

	if !ok {
		return nil
	}

	externalScope := externalNS.GetRootScope()
	return externalScope.Lookup(sym.GetName())
}

func (n *Namespace) GetRootScope() IScope {
	return &n.rootScope
}

func (n *Namespace) GetName() string {
	return n.name
}

func (n *Namespace) hasExternal(nsName string) bool {
	_, ok := n.externals[nsName]
	return ok
}

func (n *Namespace) setExternal(name string, ns *Namespace) {
	n.externals[name] = ns
}

func (n *Namespace) setForms(block *Block) {
	n.forms = *block
}

func (n *Namespace) getForms() *Block {
	return &n.forms
}

func MakeNS(name string, source string) Namespace {
	s := MakeScope(nil)
	return Namespace{
		name:      name,
		rootScope: *s,
		source:    source,
		externals: map[string]*Namespace{},
	}
}
