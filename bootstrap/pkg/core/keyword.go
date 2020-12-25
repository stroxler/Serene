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

// Keyword implementation:
// IMPORTANT NOTE: This implementation keyword is not decent at all
// it lacks many aspects of keywords which makes them great. But
// it is good enough for our use case. So we'll leave it as it is.
//
// Keywords are simple names and just names and nothing more. They
// match the following grammar:
//
// ```
// KEYWORD = COLON [COLON] SYMBOL
// ```
// Normally keywords doesn't have any namespace for example `:xyz`
// is just a name, but in order to avoid any name collision Serene
// supports namespace qualified keywords which basically put a keyword
// under a namespace. But it doesn't mean that you need to load a
// namespace in order to use any keyword that lives under that ns.
// because keywords are just names remember? There is two ways to
// use namespace qualified keywords:
//
// 1. Using full namespace name. For example, `:serene.core/xyz`
// To use this keyword you don't need the namespace `serene.core`
// to be loaded. It's just a name after all.
//
// 2. Using aliased namespaces. For example `::xyz` or `::core/xyz`.
// Using two colons instructs Serene to use aliased namespaces
// with the keyword. In the `::xyz` since the ns part is missing
// Serene will use the current namespace. For instance:
//
// ```
// user> ::xyz
// :user/xyz
// ```
// As you can see `::xyz` and `:user/xyz` (`user` being the ns name)
// are literally the same.
//
// But if we provide the ns part (`::core/xyz` example), a namespace
// with that alias has to be loaded and present in the current
// namespace. For example:
//
// ```
// user> (require '(examples.hello-world hello))
// <ns: examples.hello-world at /home/lxsameer/src/serene/serene/bootstrap/examples/hello-world.srn>
// user> ::hello/xyz
// :examples.hello-world/xyz
// ```
// As you can see we had to load the ns with the `hello` alias to be
// able to use the alias in a keyword.
//
// TODO: Cache the keywords in the runtime on the first eval so we
// done have to evaluate them over and over agian. It can be achieved
// by caching the `hash` value in the keyword itself and maintain a
// hashmap in the runtime from hash codes to a pointer to the keyword.
// But garbage collecting it would be an issue since Golang doesn't support
// weak pointer, but since bootstrap version of Serene is used only to
// bootstrap the compiler it's ok to ignore that for now

import (
	"fmt"
	"strings"

	"serene-lang.org/bootstrap/pkg/ast"
	"serene-lang.org/bootstrap/pkg/hash"
)

type Keyword struct {
	Node
	ExecutionScope
	name string
	// nsName is the string that is used as the namespace name. It
	// might be an ns alias in the current ns or the full namespace
	// as well. The first time that this keyword gets evaluated the
	// `ns` field will be populated by a pointer to the actual
	// namespace which is referrenced to via `nsName` and after that
	// nsName will be pretty much useless.
	nsName string
	// It will be populated after the first evaluation of this keyword
	ns *Namespace

	// Is it like :serene.core/something
	nsQualified bool

	// Is it like ::s/something ?
	aliased bool
}

func (k *Keyword) GetType() ast.NodeType {
	return ast.Keyword
}

func (k *Keyword) String() string {
	if k.nsQualified {
		if k.ns == nil {
			return ":" + k.nsName + "/" + k.name
		}
		return ":" + k.ns.GetName() + "/" + k.name
	}
	return ":" + k.name
}

func (k *Keyword) ToDebugStr() string {
	var ns string
	if k.nsQualified {
		ns = k.ns.GetName() + "/"
	} else {
		ns = ""
	}
	return fmt.Sprintf("<keword :%s%s at %p>", ns, k.name, k)
}

func (k *Keyword) Hash() uint32 {
	bytes := []byte(k.name)
	nameHash := hash.HashOf(append([]byte{byte(ast.Keyword)}, bytes...))

	if k.nsQualified {
		if k.ns != nil {
			return hash.CombineHashes(hash.HashOf([]byte(k.ns.GetName())), nameHash)
		}
	}

	return nameHash
}

func (k *Keyword) SetNS(ns *Namespace) {
	k.ns = ns
}

func (k *Keyword) IsNSQualified() bool {
	return k.nsQualified
}

// Eval initializes the keyword by looking up the possible
// alias name and set it in the keyword.
func (k *Keyword) Eval(rt *Runtime, scope IScope) (*Keyword, IError) {
	if k.nsQualified && k.aliased {
		aliasedNS := rt.CurrentNS()

		if k.nsName != "" {
			aliasedNS = rt.CurrentNS().LookupExternal(k.nsName)
		}

		if aliasedNS == nil {
			return nil, MakeErrorFor(rt, k, fmt.Sprintf("can't find the alias '%s' in the current namespace.", k.nsName))
		}
		k.ns = aliasedNS
		return k, nil
	}

	return k, nil
}

// Extracts the different parts of the keyword
func extractParts(s string) (string, string) {
	parts := strings.Split(s, "/")

	if len(parts) == 2 {
		return parts[0], parts[1]
	}

	return "", parts[0]
}

func MakeKeyword(n Node, name string) (*Keyword, IError) {
	if strings.Count(name, ":") > 2 {
		return nil, MakeParsetimeErrorf(n, "can't parse the keyword with more that two colons: '%s'", name)
	}

	if strings.Count(name, "/") > 1 {
		return nil, MakeParsetimeErrorf(n, "illegal namespace path for the given keyword: '%s'", name)
	}

	var nsName string
	var kwName string
	keyword := name

	nsQualified := false
	aliased := false

	if strings.HasPrefix(name, "::") {
		nsQualified = true
		aliased = true
		keyword = name[2:]
	} else if strings.HasPrefix(name, ":") && strings.Count(name, "/") == 1 {
		nsQualified = true
		keyword = name[1:]
	} else if strings.HasPrefix(name, ":") {
		keyword = name[1:]
	}

	nsName, kwName = extractParts(keyword)

	return &Keyword{
		Node:        n,
		name:        kwName,
		nsName:      nsName,
		nsQualified: nsQualified,
		aliased:     aliased,
	}, nil
}
