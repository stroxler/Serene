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
	"strings"
)

func toRepresanbleString(ast ...IRepresentable) string {
	var results []string
	for _, x := range ast {
		results = append(results, x.String())

	}
	return strings.Join(results, " ")
}

func toPrintableString(ast ...IRepresentable) string {
	var results []string
	for _, x := range ast {

		if printable, ok := x.(IPrintable); ok {
			results = append(results, printable.PrintToString())
			continue
		}
		results = append(results, x.String())

	}
	return strings.Join(results, " ")
}

func Pr(rt *Runtime, ast ...IRepresentable) {
	fmt.Print(toRepresanbleString(ast...))
}

func Prn(rt *Runtime, ast ...IRepresentable) {
	fmt.Println(toRepresanbleString(ast...))
}

func Print(rt *Runtime, ast ...IRepresentable) {
	fmt.Print(toPrintableString(ast...))
}

func Println(rt *Runtime, ast ...IRepresentable) {
	fmt.Println(toPrintableString(ast...))
}

func PrintError(rt *Runtime, err IError) {
	loc := err.GetLocation()
	fmt.Printf("Error: %s\nAt: %d to %d\n", err.String(), loc.GetStart(), loc.GetEnd())
}
