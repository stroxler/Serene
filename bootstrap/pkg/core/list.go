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
	"strings"

	"serene-lang.org/bootstrap/pkg/ast"
	"serene-lang.org/bootstrap/pkg/hash"
)

/** WARNING:
This List implementation may look simple and performant but since
we're using a slice here. But in fact it's not memory effecient at
all. We need to rewrite this later to be a immutable and persistent
link list of cons.
*/

type List struct {
	Node
	ExecutionScope
	exprs []IExpr
}

// Implementing IExpr for List ---

func (l *List) GetType() ast.NodeType {
	return ast.List
}

func (l *List) String() string {
	var strs []string
	for _, e := range l.exprs {
		strs = append(strs, e.String())
	}
	return fmt.Sprintf("(%s)", strings.Join(strs, " "))
}

func (l *List) ToDebugStr() string {
	return fmt.Sprintf("%#v", l)
}

// END: IExpr ---

// Implementing ISeq for List ---

func (l *List) First() IExpr {
	if l.Count() == 0 {
		return MakeNil(MakeNodeFromExpr(l))
	}
	return l.exprs[0]
}

func (l *List) Rest() ISeq {
	if l.Count() < 2 {
		return MakeEmptyList(l.Node)
	}

	rest := l.exprs[1:]
	node := l.Node
	if len(rest) > 0 {
		// n won't be nil here but we should check anyway
		n := MakeNodeFromExprs(rest)
		if n == nil {
			panic("'MakeNodeFromExprs' has returned nil for none empty array of exprs")
		}
		node = *n
	}

	return MakeList(node, rest)
}

func (l *List) Hash() uint32 {
	bytes := []byte("TODO")
	return hash.HashOf(append([]byte{byte(ast.List)}, bytes...))
}

// END: ISeq ---

// Implementing ICountable for List ---

func (l *List) Count() int {
	return len(l.exprs)
}

// END: ICountable ---

// Implementing IColl for List ---

func (l *List) ToSlice() []IExpr {
	return l.exprs
}

func (l *List) Cons(e IExpr) IExpr {
	elements := append([]IExpr{e}, l.ToSlice()...)
	node := MakeNodeFromExprs(elements)

	// Since 'elements' is not empty node won't be nil but we should
	// check anyway
	if node == nil {
		node = &l.Node
	}
	return MakeList(*node, elements)
}

// END: IColl ---

func (l *List) AppendToList(e IExpr) *List {
	l.exprs = append(l.exprs, e)
	return l
}

func ListStartsWith(l *List, sym string) bool {
	if l.Count() > 0 {
		firstElem := l.First()
		if firstElem.GetType() == ast.Symbol {
			return firstElem.(*Symbol).GetName() == sym
		}
	}
	return false
}

func MakeList(n Node, elements []IExpr) *List {
	return &List{
		Node:  n,
		exprs: elements,
	}
}

func MakeEmptyList(n Node) *List {
	return &List{
		Node:  n,
		exprs: []IExpr{},
	}
}
