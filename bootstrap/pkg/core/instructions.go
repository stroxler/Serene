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

// Instructions Implementation:
// * Instructions are expressions as well but they are not doing
//   anything special by themselves.
// * We need them to be expressions because we need to process
//   them as part of the evaluation loop
// * We use instructions as nodes in the AST to instruct Serene
//   to do specific tasks after rewriting the AST. For example
//   `PopStack` instructs Serene to simply pop a call from the
//   call stack.
// * Instructions doesn't return a value and should not alter
//   the return value of the eval loop. But they might interrupt
//   the loop by raising an error `IError`.
import (
	"fmt"

	"serene-lang.org/bootstrap/pkg/ast"
	"serene-lang.org/bootstrap/pkg/hash"
)

type instructionType int

// Instruction types
const (
	// Pop a function from the call stack. We use this
	// instruction at the end of function bodies. So
	// function bodies will clean up after themselves
	PopStack instructionType = iota
)

type Instruction struct {
	// Just to be compatible with IExpr ---
	Node
	ExecutionScope
	// ------------------------------------

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

// ProcessInstruction is the main function to process instructions
func ProcessInstruction(rt *Runtime, form *Instruction) IError {
	switch form.Type {
	case PopStack:
		rt.Stack.Pop()
		return nil
	default:
		panic(fmt.Sprintf("Unknown instruction: '%d'", form.Type))
	}
}
