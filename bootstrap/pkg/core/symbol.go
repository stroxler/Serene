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

type Symbol struct {
	Node
	name   string
	nsPart string
}

func (s *Symbol) GetType() ast.NodeType {
	return ast.Symbol
}

func (s *Symbol) String() string {
	if s.IsNSQualified() {
		return s.nsPart + "/" + s.name
	}

	return s.name
}

func (s *Symbol) GetName() string {
	return s.name
}

func (s *Symbol) GetNSPart() string {
	return s.nsPart
}

func (s *Symbol) ToDebugStr() string {
	return s.String()
}

func (s *Symbol) IsRestable() bool {
	// Weird name ? I know :D
	return strings.HasPrefix(s.name, "&")
}

func (s *Symbol) IsNSQualified() bool {
	if s.nsPart == "" {
		return false
	}
	return true
}

func MakeSymbol(n Node, s string) (*Symbol, IError) {
	parts := strings.Split(s, "/")
	var (
		name   string
		nsPart string
	)

	switch len(parts) {
	case 1:
		name = parts[0]
		nsPart = ""
	case 2:
		name = parts[1]
		nsPart = parts[0]
	default:
		return nil, MakePlainError(fmt.Sprintf("can't create a symbol from '%s'. More that on '/' is illegal.", s))
	}

	return &Symbol{
		Node:   n,
		name:   name,
		nsPart: nsPart,
	}, nil
}
