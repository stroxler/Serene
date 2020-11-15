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

// Package core contains the high level internal function of Serene
package core

import (
	"fmt"

	"github.com/chzyer/readline"
	"serene-lang.org/bootstrap/pkg/printer"
	"serene-lang.org/bootstrap/pkg/reader"
	"serene-lang.org/bootstrap/pkg/runtime"
)

func rep(rt *runtime.Runtime, line string) {
	ast, err := reader.ReadString(line)

	if err != nil {
		fmt.Println(err)
	}

	if rt.IsDebugMode() {
		fmt.Println("\n### DEBUG ###")
		printer.Print(rt, ast)
		fmt.Println("#############\n")
	}

	result, err := Eval(rt, ast)
	if err != nil {
		fmt.Printf("Error: %s\n", err)
		return
	}

	printer.Print(rt, result)
}

/** TODO:
Replace the readline implementation with go-prompt.
*/

func REPL(debug bool) {
	rt := runtime.MakeRuntime(debug)

	rt.CreateNS("user", "REPL", true)
	rl, err := readline.New("> ")
	if err != nil {
		panic(err)
	}
	defer rl.Close()

	fmt.Println(`Serene's bootstrap interpreter is used to
bootstrap the Serene's compiler.'

It comes with ABSOLUTELY NO WARRANTY;
This is free software, and you are welcome
to redistribute it under certain conditions;
for details take a look at the LICENSE file.
`)
	for {
		rl.SetPrompt(fmt.Sprintf("%s> ", rt.CurrentNS().GetName()))
		line, err := rl.Readline()
		if err != nil { // io.EOF
			break
		}
		rep(rt, line)
	}

}
