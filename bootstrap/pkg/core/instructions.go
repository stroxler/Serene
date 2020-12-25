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

type instructionType int

const (
	PopStack instructionType = iota
)

type Instruction struct {
	Node
	ExecutionScope
	Type instructionType
}

func (n *Instruction) GetType() ast.NodeType {
	return ast.Instruction
}

func (n *Instruction) String() string {
	return fmt.Sprintf("<instruction:%d>", n.Type)
}

func (n *Instruction) ToDebugStr() string {
	return n.String()
}

func (n *Instruction) Hash() uint32 {
	bytes := []byte(fmt.Sprintf("%d", n.Type))
	return hash.HashOf(append([]byte{byte(ast.Instruction)}, bytes...))
}

func MakeStackPop(rt *Runtime) IExpr {
	return &Instruction{
		Type: PopStack,
	}
}

func ProcessInstruction(rt *Runtime, form *Instruction) IError {
	switch form.Type {
	case PopStack:
		rt.Stack.Pop()
		return nil
	default:
		panic(fmt.Sprintf("Unknown instruction: '%d'", form.Type))
	}
}
