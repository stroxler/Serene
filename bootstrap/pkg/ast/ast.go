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

// Package ast provides the functionality and data structures around the
// Serene's AST.
package ast

type NodeType int

const (
	Nil NodeType = iota
	Nothing
	True
	False
	Instruction
	Symbol
	Keyword
	Number
	List
	Fn
	NativeFn
	Namespace
	String
	Block // Dont' mistake it with block from other programming languages

)

type Location struct {
	start         int
	end           int
	source        *[]string
	knownLocation bool
}

func (l *Location) GetStart() int {
	return l.start
}

func (l *Location) GetEnd() int {
	return l.end
}

func (l *Location) GetSource() *[]string {
	return l.source
}

func (l *Location) IsKnownLocaiton() bool {
	return l.knownLocation
}

type ILocatable interface {
	GetLocation() Location
}

func MakeLocation(input *[]string, start int, end int) Location {
	return Location{
		source:        input,
		start:         start,
		end:           end,
		knownLocation: true,
	}
}

type ITypable interface {
	GetType() NodeType
}

func MakeUnknownLocation() Location {
	return Location{
		knownLocation: false,
	}
}
