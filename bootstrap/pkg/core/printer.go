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
	"serene-lang.org/bootstrap/pkg/ast"
	"serene-lang.org/bootstrap/pkg/errors"
)

func toRepresanbleString(forms ...IRepresentable) string {
	var results []string

	for _, x := range forms {
		results = append(results, x.String())
	}

	return strings.Join(results, " ")
}

func toPrintableString(forms ...IRepresentable) string {
	var results []string

	for _, x := range forms {
		if printable, ok := x.(IPrintable); ok {
			results = append(results, printable.PrintToString())
			continue
		}
		results = append(results, x.String())
	}

	return strings.Join(results, " ")
}

func Pr(rt *Runtime, forms ...IRepresentable) {
	fmt.Print(toRepresanbleString(forms...))
}

func Prn(rt *Runtime, forms ...IRepresentable) {
	fmt.Println(toRepresanbleString(forms...))
}

func Print(rt *Runtime, forms ...IRepresentable) {
	fmt.Print(toPrintableString(forms...))
}

func Println(rt *Runtime, forms ...IRepresentable) {
	fmt.Println(toPrintableString(forms...))
}

func printError(_ *Runtime, err IError, stage int) {
	loc := err.GetLocation()
	source := loc.GetSource()

	startline := source.LineNumberFor(loc.GetStart())

	if startline > 0 {
		startline--
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
		"%d: At '%s':%d\n",
		stage,
		source.NS,
		source.LineNumberFor(loc.GetStart()),
	)

	color.White.Printf("%s\n", lines)

	errTag := color.Red.Sprint(err.GetErrType().String())
	fmt.Printf("%s: %s\nAt: %d to %d\n", errTag, err.String(), loc.GetStart(), loc.GetEnd())
}

func frameCaption(traces *TraceBack, frameIndex int) string {
	if frameIndex >= len(*traces) || frameIndex < 0 {
		panic("Out of range index for the traceback array. It shouldn't happen!!!")
	}
	var prevFrame *Frame
	place := "*run*"
	frame := (*traces)[frameIndex]

	loc := frame.Caller.GetLocation()
	source := loc.GetSource()

	if frameIndex != 0 {
		prevFrame = (*traces)[frameIndex-1]
		place = prevFrame.Callee.GetName()
	}

	return color.Yellow.Sprintf(
		"%d: In function '%s' at '%s':%d\n",
		frameIndex,
		place,
		source.NS,
		source.LineNumberFor(loc.GetStart()),
	)
}

func frameSource(traces *TraceBack, frameIndex int) string {
	if frameIndex >= len(*traces) || frameIndex < 0 {
		panic("Out of range index for the traceback array. It shouldn't happen!!!")
	}

	frame := (*traces)[frameIndex]
	caller := frame.Caller
	callerLoc := caller.GetLocation()
	callerSource := callerLoc.GetSource()

	startline := callerSource.LineNumberFor(callerLoc.GetStart())

	if startline > 0 {
		startline--
	}

	endline := callerSource.LineNumberFor(callerLoc.GetEnd()) + 1

	var lines string
	for i := startline; i <= endline; i++ {
		fLoc := frame.Caller.GetLocation()

		if fLoc.IsKnownLocaiton() {
			line := fLoc.GetSource().GetLine(i)
			if line != ast.OutOfRangeLine {
				lines += fmt.Sprintf("%d:\t%s\n", i, line)
			}
		} else {
			lines += "Builtin\n"
		}
	}

	return lines
}

func printErrorWithTraceBack(_ *Runtime, err IError) {
	trace := err.GetStackTrace()

	for i := range *trace {
		fmt.Print(frameCaption(trace, i))
		color.White.Printf(frameSource(trace, i))
	}
	loc := err.GetLocation()
	errTag := color.Red.Sprint(err.GetErrType().String())
	fmt.Printf(
		"%s: %s\nAt: %d to %d\n",
		errTag,
		err.String(),
		loc.GetStart(),
		loc.GetEnd(),
	)
	if err.GetErrno() != errors.E0000 {
		fmt.Printf("For more information on this error try: `serene explain %s`\n", err.GetErrno())
	}
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
