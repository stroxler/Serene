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
)

type IError interface {
	ast.ILocatable
	IPrintable
	IDebuggable
}

type Error struct {
	Node
	msg string
}

func (e *Error) String() string {
	return e.msg
}

func (e *Error) ToDebugStr() string {
	return e.msg
}

func MakeError(rt *Runtime, msg string) IError {
	return &Error{
		msg: msg,
	}
}

func MakeRuntimeErrorf(rt *Runtime, msg string, a ...interface{}) IError {
	return &Error{
		msg: fmt.Sprintf(msg, a...),
	}
}

func MakeParsetimeErrorf(n Node, msg string, a ...interface{}) IError {
	return &Error{
		Node: n,
		msg:  fmt.Sprintf(msg, a...),
	}
}
