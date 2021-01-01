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

import (
	"sort"
	"strings"
)

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

type Source struct {
	Buffer *[]string
	// It can be the path to the source file or something like "*in*"
	// for standard in
	Path      string
	LineIndex *[]int
}

func (s *Source) GetPos(start, end int) *string {
	if start < len(*s.Buffer) && start >= 0 && end < len(*s.Buffer) && end > 0 && start <= end {
		result := strings.Join((*s.Buffer)[start:end], "")
		return &result
	} else {
		return nil
	}
}
func (s *Source) GetLine(linenum int) string {
	lines := strings.Split(strings.Join(*s.Buffer, ""), "\n")
	if linenum > 0 && linenum < len(lines) {
		return lines[linenum-1]
	}
	return "!!!"
}

func (s *Source) LineNumberFor(pos int) int {

	// Some dirty print debugger code
	// for i, r := range *s.LineIndex {
	// 	empty := ""
	// 	var line *string
	// 	var num int
	// 	if i == 0 {
	// 		line = s.GetPos(0, r)
	// 		num = 0
	// 	} else {
	// 		line = s.GetPos((*s.LineIndex)[i-1], r)
	// 		num = (*s.LineIndex)[i-1]

	// 	}

	// 	if line == nil {
	// 		line = &empty
	// 	}

	// 	fmt.Print(">>>> ", num, r, *line)
	// }

	if pos < 0 {
		return -1
	}

	result := sort.Search(len(*s.LineIndex), func(i int) bool {
		if i == 0 {
			return pos < (*s.LineIndex)[i]
		} else {
			return (*s.LineIndex)[i-1] < pos && pos < (*s.LineIndex)[i]
		}
	})

	// We've found something
	if result > -1 {
		// Since line numbers start from 1 unlike arrays :))
		result += 1
	}

	return result

}

type Location struct {
	start         int
	end           int
	source        Source
	knownLocation bool
}

var UnknownLocation *Location = &Location{knownLocation: false}

func (l *Location) GetStart() int {
	return l.start
}

func (l *Location) GetEnd() int {
	return l.end
}

func (l *Location) GetSource() *Source {
	return &l.source
}

func (l *Location) IncStart(x int) {
	if x+l.start < len(*l.source.Buffer) {
		l.start += x
	} else {
		l.start = len(*l.source.Buffer) - 1
	}
}

func (l *Location) DecStart(x int) {
	if l.start-x >= 0 {
		l.start -= x
	} else {
		l.start = 0
	}

}

func (l *Location) IncEnd(x int) {
	if x+l.end < len(*l.source.Buffer) {
		l.end += x
	} else {
		l.end = len(*l.source.Buffer) - 1
	}

}

func (l *Location) DecEnd(x int) {
	if l.end-x >= 0 {
		l.end -= x
	} else {
		l.end = 0
	}
}

func (l *Location) IsKnownLocaiton() bool {
	return l.knownLocation
}

type ILocatable interface {
	GetLocation() *Location
}

func MakeLocation(input *Source, start int, end int) *Location {
	return &Location{
		source:        *input,
		start:         start,
		end:           end,
		knownLocation: true,
	}
}

type ITypable interface {
	GetType() NodeType
}

func MakeUnknownLocation() *Location {
	return UnknownLocation
}
