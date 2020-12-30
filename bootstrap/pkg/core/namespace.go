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
	"serene-lang.org/bootstrap/pkg/hash"
)

type INamespace interface {
	// TODO: Add a method to fetch the source code based on the source value
	DefineGlobal()
	LookupGlobal()
	GetRootScope() IScope
	// return the fully qualified name of the namespace
	GetName() string
	getForms() *Block
	setForms(forms *Block)
}

type Namespace struct {
	// Fully qualified name of the ns. e.g `serene.core`
	name string

	// The root scope of the namespace which keeps the global definitions
	// and values of the scope. But not the external namespaces and values
	rootScope Scope

	// TODO: Add a method to fetch the source code based on this value
	// Path to the source of the name space, it can be a file path
	// or anything related to the ns loader (not implemented yet).
	source    string
	externals map[string]*Namespace
	forms     Block
}

func (n *Namespace) GetType() ast.NodeType {
	return ast.Namespace
}

func (n *Namespace) GetLocation() *ast.Location {
	return ast.MakeUnknownLocation()
}

func (n *Namespace) String() string {
	return fmt.Sprintf("<ns: %s at %s>", n.name, n.source)
}

func (n *Namespace) ToDebugStr() string {
	return fmt.Sprintf("<ns: %s at %s>", n.name, n.source)
}

func (n *Namespace) GetExecutionScope() IScope {
	return nil
}

func (n *Namespace) SetExecutionScope(scope IScope) {}

// DefineGlobal inserts the given expr `v` to the root scope of
// `n`. The `public` parameter determines whether the public
// value is accessable publicly or not (in other namespaces).
func (n *Namespace) DefineGlobal(k string, v IExpr, public bool) {
	n.rootScope.Insert(k, v, public)
}

// LookupGlobal looks up the value represented by the ns qualified
// symbol `sym` in the external symbols table. Simply looking up
// a public value from an external namespace.
func (n *Namespace) LookupGlobal(rt *Runtime, sym *Symbol) *Binding {
	// TODO: Find a better name for this method, `LookupExternal` maybe
	if !sym.IsNSQualified() {
		return nil
	}

	externalNS, ok := n.externals[sym.GetNSPart()]

	if !ok {
		return nil
	}

	externalScope := externalNS.GetRootScope()
	return externalScope.Lookup(rt, sym.GetName())
}

func (n *Namespace) GetRootScope() IScope {
	return &n.rootScope
}

func (n *Namespace) GetName() string {
	return n.name
}

func (n *Namespace) Hash() uint32 {
	return hash.HashOf([]byte(n.String()))
}

func (n *Namespace) hasExternal(nsName string) bool {
	_, ok := n.externals[nsName]
	return ok
}

// LookupExternal looks up the given `alias` in the `externals` table
// of the namespace.
func (n *Namespace) LookupExternal(alias string) *Namespace {
	if n.hasExternal(alias) {
		return n.externals[alias]
	}

	return nil
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

// requireNS finds and loads the namespace addressed by the given
// `ns` string.
func requireNS(rt *Runtime, ns *Symbol) (*Namespace, IError) {
	// TODO: use a hashing algorithm to avoid reloading an unchanged namespace
	loadedForms, err := rt.LoadNS(ns)

	if err != nil {
		return nil, err
	}

	body := loadedForms.forms
	source := loadedForms.source

	if body.Count() == 0 {
		return nil, MakeError(
			rt,
			body,
			fmt.Sprintf("The '%s' ns source code doesn't start with an 'ns' form.", ns),
		)
	}

	namespace := MakeNS(ns.GetName(), source)
	namespace.setForms(body)

	return &namespace, nil
}

// RequireNamespace finds and loads the naemspace which is addressed by the
// given expression `namespace` and add the loaded namespace to the list
// of available namespaces on the runtime and add the correct reference
// to the current namespace. If `namespace` is a symbol, then the name
// of the symbol would be used as the ns name and an alias with the
// same name will be added to the current namespace as the external
// reference. If it is a IColl (List at the moment), then the symbol
// in the first element would be the ns name and the second symbol
// will be the name of the alias to be used.
func RequireNamespace(rt *Runtime, namespace IExpr) (IExpr, IError) {
	var alias string
	var ns *Symbol

	switch namespace.GetType() {
	case ast.Symbol:
		ns = namespace.(*Symbol)
		alias = ns.GetName()

	case ast.List:
		list := namespace.(*List)
		first := list.First()

		if first.GetType() != ast.Symbol {
			return nil, MakeError(rt, first, "The first element has to be a symbol")
		}

		second := list.Rest().First()
		if second.GetType() != ast.Symbol {
			return nil, MakeError(rt, first, "The second element has to be a symbol")
		}

		ns = first.(*Symbol)
		alias = second.(*Symbol).GetName()
	default:
		return nil, MakeError(rt, ns, "Don't know how to load the given namespace")
	}

	loadedNS, err := requireNS(rt, ns)

	if err != nil {
		return nil, err
	}

	// Since we want to change the current ns to the loaded ns while evaluating it.
	prevNS := rt.CurrentNS()

	rt.InsertNS(ns.GetName(), loadedNS)
	inserted := rt.setCurrentNS(loadedNS.GetName())

	if !inserted {
		return nil, MakeError(
			rt,
			loadedNS,
			fmt.Sprintf(
				"the namespace '%s' didn't get inserted in the runtime.",
				loadedNS.GetName()),
		)
	}

	// Evaluating the body of the loaded ns (Check for ns validation happens here)
	loadedNS, e := EvalNSBody(rt, loadedNS)

	// Set the current ns back first and then check for an error
	inserted = rt.setCurrentNS(prevNS.GetName())
	if !inserted {
		return nil, MakeError(
			rt,
			loadedNS,
			fmt.Sprintf(
				"can't set the current ns back to '%s' from '%s'.",
				prevNS.GetName(),
				loadedNS.GetName()),
		)
	}

	if e != nil {
		return nil, e
	}

	// Set the external reference to the loaded ns in the current ns
	prevNS.setExternal(alias, loadedNS)
	return loadedNS, nil
}

// MakeNS creates a new namespace with the given `name` and `source`
func MakeNS(name string, source string) Namespace {
	s := MakeScope(nil)
	return Namespace{
		name:      name,
		rootScope: *s,
		source:    source,
		externals: map[string]*Namespace{},
	}
}
