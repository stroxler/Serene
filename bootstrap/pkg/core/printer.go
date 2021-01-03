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

	"github.com/gookit/color"
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

func printError(rt *Runtime, err IError, stage int) {
	loc := err.GetLocation()
	source := loc.GetSource()

	startline := source.LineNumberFor(loc.GetStart())

	if startline > 0 {
		startline -= 1
	}

	endline := source.LineNumberFor(loc.GetEnd()) + 1

	var lines string
	for i := startline; i <= endline; i++ {
		line := source.GetLine(i)
		if line != "----" {
			lines += fmt.Sprintf("%d:\t%s\n", i, line)
		}
	}

	color.Yellow.Printf(
		"%d: At '%s'\n",
		stage,
		source.Path,
	)

	color.White.Printf("%s\n", lines)

	errTag := color.Red.Sprint(err.GetErrType().String())
	fmt.Printf("%s: %s\nAt: %d to %d\n", errTag, err.String(), loc.GetStart(), loc.GetEnd())

}

func printErrorWithTraceBack(rt *Runtime, err IError) {
	trace := err.GetStackTrace()

	for i, t := range *trace {
		caller := t.Caller
		callerLoc := caller.GetLocation()
		callerSource := callerLoc.GetSource()

		startline := callerSource.LineNumberFor(callerLoc.GetStart())

		if startline > 0 {
			startline -= 1
		}

		endline := callerSource.LineNumberFor(callerLoc.GetEnd()) + 1

		var lines string
		for i := startline; i <= endline; i++ {
			fLoc := t.Fn.GetLocation()
			line := fLoc.GetSource().GetLine(i)
			if line != "----" {
				lines += fmt.Sprintf("%d:\t%s\n", i, line)
			}
		}

		color.Yellow.Printf(
			"%d: In function '%s' at '%s':%d\n",
			i,
			t.Fn.GetName(),
			callerLoc.GetSource().Path,
			callerSource.LineNumberFor(callerLoc.GetStart()),
		)
		color.White.Printf("%s\n", lines)
	}
	loc := err.GetLocation()
	errTag := color.Red.Sprint(err.GetErrType().String())
	fmt.Printf("%s: %s\nAt: %d to %d\n", errTag, err.String(), loc.GetStart(), loc.GetEnd())
}

func PrintError(rt *Runtime, err IError) {
	switch err.GetErrType() {
	case SyntaxError, SemanticError:
		printError(rt, err, 0)
		return
	case RuntimeError:
		printErrorWithTraceBack(rt, err)
		return
	default:
		panic(fmt.Sprintf("Don't know about error type '%d'", err.GetErrType()))
	}
}
