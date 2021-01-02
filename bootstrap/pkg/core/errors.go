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

// Error implementations:
// * `IError` is the main interface to represent errors.
// * `Error` struct is an expression itself.
// * `IError` and any implementation of it has to implement `ILocatable`
//    so we can point to the exact location of the error.
// * We have to use `IError` everywhere and avoid using Golangs errors
//   since IError is an expression itself.
//
// TODOs:
// * Make errors stackable, so different pieces of code can stack related
//   errors on top of each other so user can track them through the code
// * Errors should contain a help message as well to give some hints to the
//   user about how to fix the problem. Something similar to Rust's error
//   messages
// * Integrate the call stack with IError

import (
	"fmt"

	"serene-lang.org/bootstrap/pkg/ast"
	"serene-lang.org/bootstrap/pkg/errors"
)

type ErrType uint8

const (
	SyntaxError ErrType = iota
	SemanticError
	RuntimeError
)

// IError defines the necessary functionality of the internal errors.
type IError interface {
	// In order to point to a specific point in the input
	ast.ILocatable

	// We want errors to be printable by the `print` family
	IRepresentable
	IDebuggable

	GetErrType() ErrType
	GetDescription() *string
	GetStackTrace() *TraceBack
	// To wrap Golan rrrrors
	WithError(err error) IError

	// Some errors might doesn't have any node available to them
	// at the creation time. SetNode allows us to the the appropriate
	// node later in time.
	SetNode(n *Node)
}

type Error struct {
	Node
	errtype    ErrType
	errno      errors.Errno
	WrappedErr error
	msg        string
	trace      *TraceBack
}

func (e *Error) String() string {
	return e.msg
}

func (e *Error) ToDebugStr() string {
	_, isInternalErr := e.WrappedErr.(*Error)
	if isInternalErr {
		return fmt.Sprintf("%s:\n\t%s", e.msg, e.WrappedErr.(*Error).ToDebugStr())
	}
	return fmt.Sprintf("%s:\n\t%s", e.msg, e.WrappedErr.Error())
}

func (e *Error) GetErrType() ErrType {
	return e.errtype
}

func (e *Error) WithError(err error) IError {
	e.WrappedErr = err
	return e
}

func (e *Error) SetNode(n *Node) {
	e.Node = *n
}

func (e *Error) Error() string {
	return e.msg
}

func (e *Error) GetStackTrace() *TraceBack {
	return e.trace
}

func (e *Error) GetDescription() *string {
	desc, ok := errors.ErrorsDescription[e.errno]
	if ok {
		return &desc
	}

	desc = errors.ErrorsDescription[0]
	return &desc
}

func MakePlainError(msg string) IError {
	return &Error{
		msg: msg,
	}
}

// MakeError creates an Error which points to the given IExpr `e` as
// the root of the error.
func MakeError(rt *Runtime, e IExpr, msg string) IError {
	frame := MakeFrame(e, rt.Stack.GetCurrentFn(), 1)
	trace := append(*rt.Stack.ToTraceBack(), frame)

	return &Error{
		Node:    MakeNodeFromExpr(e),
		errtype: RuntimeError,
		msg:     msg,
		trace:   &trace,
	}
}

func MakeSyntaxErrorf(n Node, msg string, a ...interface{}) IError {
	return &Error{
		Node:    n,
		errtype: SyntaxError,
		msg:     fmt.Sprintf(msg, a...),
	}
}

func MakeSemanticError(rt *Runtime, e IExpr, errno errors.Errno, msg string) IError {
	currentFn := rt.Stack.GetCurrentFn()

	frames := &[]*Frame{
		MakeFrame(e, currentFn, 1),
	}

	return &Error{
		Node:    MakeNodeFromExpr(e),
		errtype: SemanticError,
		errno:   errno,
		msg:     msg,
		trace:   frames,
	}
}
