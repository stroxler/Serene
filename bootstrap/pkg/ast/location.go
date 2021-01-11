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
	if l.IsKnownLocaiton() {
		return &l.source
	}
	return GetBuiltinSource()
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

func MakeUnknownLocation() *Location {
	return UnknownLocation
}
