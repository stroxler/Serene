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

type String struct {
	Node
	ExecutionScope
	content string
}

func (s *String) GetType() ast.NodeType {
	return ast.String
}

func (s *String) String() string {
	return "\"" + s.Escape() + "\""
}

func (s *String) PrintToString() string {
	return s.content
}

func (s *String) ToDebugStr() string {
	return fmt.Sprintf("<%s at %p>", s.content, s)
}

func (s *String) Hash() uint32 {
	bytes := []byte(s.content)
	return hash.HashOf(append([]byte{byte(ast.String)}, bytes...))
}

func (s *String) Escape() string {
	replacer := strings.NewReplacer(
		"\n", "\\n",
		"\t", "\\t",
		"\r", "\\r",
		"\\", "\\\\",
		"\"", "\\\"",
	)
	return replacer.Replace(s.content)
}

func MakeString(n Node, s string) *String {
	return &String{
		Node:    n,
		content: s,
	}
}
