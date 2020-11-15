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
)

/** WARNING:
This List implementation may look simple and performant but since
we're using a slice here. But in fact it's not memory effecient at
all. We need to rewrite this later to be a immutable and persistent
link list of cons.
*/

type List struct {
	Node
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
		return Nil
	}
	return l.exprs[0]
}

func (l *List) Rest() *List {
	if l.Count() < 2 {
		return MakeEmptyList()
	}
	return MakeList(l.exprs[1:])
}

// END: ISeq ---

// Implementing ICountable for List ---

func (l *List) Count() int {
	return len(l.exprs)
}

// END: ICountable ---

func MakeList(elements []IExpr) *List {
	return &List{
		exprs: elements,
	}
}

func MakeEmptyList() *List {
	return &List{
		exprs: []IExpr{},
	}
}
