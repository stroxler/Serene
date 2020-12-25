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

// CallStack implementation:
// * A callstack should be FIFA stack
// * It should keep track of function calls.
// * Anything that implements `IFn` can be tracked by the call stack
// * Since Serene uses eval loop to eliminate tail calls we need
// the call stack to be able to track recursive calls. For
// now by just counting number of calls to a functions that is already
// in the stack.
//
// TODOs:
// * At the moment if we call the same function twice (not as a recursive)
//   function call stack will record it as a recursive call. We need
//   compare the stack items by their address, identity and location.
// * Add support for iteration on the stack.

import "fmt"

type ICallStack interface {
	// Push the given callable `f` to the stack
	Push(f IFn) IError
	Pop() FnCall
	Count() uint
}

type FnCall struct {
	Fn IFn
	// Number of recursive calls to this function
	count uint
}

type CallStackItem struct {
	prev *CallStackItem
	data FnCall
}

type CallStack struct {
	head  *CallStackItem
	count uint
	debug bool
}

func (c *CallStack) Count() uint {
	return c.count
}

func (c *CallStack) Push(f IFn) IError {

	if c.debug {
		fmt.Println("[Stack] -->", f)
	}

	if f == nil {
		return MakePlainError("Can't push 'nil' pointer to the call stack.")
	}

	// Empty Stack
	if c.head == nil {
		c.head = &CallStackItem{
			data: FnCall{
				Fn:    f,
				count: 0,
			},
		}
		c.count++
	}

	nodeData := &c.head.data

	// If the same function was on top of the stack
	if nodeData.Fn == f {
		// TODO: expand the check here to support address and location as well
		nodeData.count++
	} else {
		c.head = &CallStackItem{
			prev: c.head,
			data: FnCall{
				Fn:    f,
				count: 0,
			},
		}
		c.count++
	}
	return nil
}

func (c *CallStack) Pop() *FnCall {
	if c.head == nil {
		if c.debug {
			fmt.Println("[Stack] <-- nil")
		}
		return nil
	}

	result := c.head
	c.head = result.prev
	c.count--
	if c.debug {
		fmt.Printf("[Stack] <-- %s\n", result.data.Fn)
	}
	return &result.data
}

func MakeCallStack(debugMode bool) CallStack {
	return CallStack{
		count: 0,
		head:  nil,
		debug: debugMode,
	}
}
