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

package ast

// The Source data structure is used to track expression back to the source
// code. For example to find which source (file) they belongs to.

import (
	"sort"
	"strings"
)

var builtinSource *Source

type Source struct {
	// A Pointer to the buffer where the parser used for parsing the source
	Buffer *[]string

	// The namespace name which this source is describing
	NS string

	// This array contains the boundaries of each line in the buffer. For example
	// [24 50 106] means that the buffer contains 3 lines and the first line can
	// be found from the index 0 to index 24 of the buffer and the second line is
	// from index 25 till 50 and so on
	LineIndex *[]int
}

// GetSubstr returns the a pointer to the string from the buffer specified by the `start` and `end`
func (s *Source) GetSubstr(start, end int) *string {
	if start < len(*s.Buffer) && start >= 0 && end < len(*s.Buffer) && end > 0 && start <= end {
		result := strings.Join((*s.Buffer)[start:end], "")
		return &result
	}

	return nil
}

// GetLine returns the line specified by the `linenum` from the buffer. It will return "----" if the
// given line number exceeds the boundaries of the buffer
func (s *Source) GetLine(linenum int) string {
	lines := strings.Split(strings.Join(*s.Buffer, ""), "\n")
	if linenum > 0 && linenum <= len(lines) {
		return lines[linenum-1]
	}
	return "----"
}

// LineNumberFor returns the line number associated with the given position `pos` in
// the buffer
func (s *Source) LineNumberFor(pos int) int {
	if pos < 0 {
		return -1
	}

	result := sort.SearchInts(*s.LineIndex, pos)

	// We've found something
	if result > -1 {
		// Since line numbers start from 1 unlike arrays :))
		result++
	}

	return result
}

// GetBuiltinSource returns a pointer to a source that represents builtin
// expressions
func GetBuiltinSource() *Source {
	if builtinSource == nil {
		buf := strings.Split("builtin", "")
		lineindex := []int{len(buf) - 1}
		builtinSource = &Source{
			Buffer:    &buf,
			NS:        "Serene.builtins",
			LineIndex: &lineindex,
		}
	}
	return builtinSource
}
