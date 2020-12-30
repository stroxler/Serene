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

// Block struct represents a group of forms. Don't confuse it with
// code blocks from other languages that specify a block using curly
// brackets and indentation.
// Blocks in serene are just a group of forms and nothing more.
type Block struct {
	ExecutionScope
	body []IExpr
}

func (b *Block) GetType() ast.NodeType {
	return ast.Block
}

func (b *Block) String() string {
	var strs []string
	for _, e := range b.body {
		strs = append(strs, e.String())
	}
	return strings.Join(strs, " ")
}

func (b *Block) ToDebugStr() string {
	return fmt.Sprintf("%#v", b)
}

func (b *Block) GetLocation() *ast.Location {
	if len(b.body) > 0 {
		return b.body[0].GetLocation()
	}
	return ast.MakeUnknownLocation()
}

func (l *Block) Hash() uint32 {
	bytes := []byte("TODO")
	return hash.HashOf(append([]byte{byte(ast.Block)}, bytes...))
}

func (b *Block) ToSlice() []IExpr {
	return b.body
}

func (b *Block) SetContent(body []IExpr) {
	b.body = body
}

// Append the given expr `form` to the block
func (b *Block) Append(form IExpr) {
	b.body = append(b.body, form)
}

func (b *Block) Count() int {
	return len(b.body)
}

// MakeEmptyBlock creates an empty block
func MakeEmptyBlock() *Block {
	return &Block{}
}

// MakeBlock creates a block that holds the given array of
// forms `body`.
func MakeBlock(body []IExpr) *Block {
	return &Block{
		body: body,
	}
}
