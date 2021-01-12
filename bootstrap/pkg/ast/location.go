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

// ILocatable describes something that can be located in a source
type ILocatable interface {
	GetLocation() *Location
}

// Location is used to point to a specific location in the source
// code (before parse). It can point to a single point or a range.
type Location struct {
	start int
	end   int
	// Where this location is pointing too ?
	source Source

	// Is it a known location or not ? For example builtins doesn't
	// have a knowen location
	knownLocation bool
}

var UnknownLocation *Location = &Location{knownLocation: false}

func (l *Location) GetStart() int {
	return l.start
}

func (l *Location) GetEnd() int {
	return l.end
}

// GetSource returns the source of the current location or the "builtin"
// source to indicate that the source of this location is not known in
// the context of source code
func (l *Location) GetSource() *Source {
	if l.IsKnownLocaiton() {
		return &l.source
	}
	return GetBuiltinSource()
}

// IncStart increases the start pointer of the location by `x` with respect
// to the boundaries of the source
func (l *Location) IncStart(x int) {
	if x+l.start < len(*l.source.Buffer) {
		l.start += x
	} else {
		l.start = len(*l.source.Buffer) - 1
	}
}

// DecStart decreases the start pointer of the location by `x` with respect
// to the boundaries of the source
func (l *Location) DecStart(x int) {
	if l.start-x >= 0 {
		l.start -= x
	} else {
		l.start = 0
	}

}

// IncEnd increases the end pointer of the location by `x` with respect
// to the boundaries of the source
func (l *Location) IncEnd(x int) {
	if x+l.end < len(*l.source.Buffer) {
		l.end += x
	} else {
		l.end = len(*l.source.Buffer) - 1
	}

}

// DecEnd decreases the end pointer of the location by `x` with respect
// to the boundaries of the source
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

// MakeLocation returns a pointer to a `Location` in the given source `input`
// specified by the `start` and `end` boundaries
func MakeLocation(input *Source, start int, end int) *Location {
	return &Location{
		source:        *input,
		start:         start,
		end:           end,
		knownLocation: true,
	}
}

func MakeUnknownLocation() *Location {
	return UnknownLocation
}
