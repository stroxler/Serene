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
	"os"
	"path/filepath"

	"github.com/chzyer/readline"
)

func rep(rt *Runtime, line string) {
	ast, err := ReadString(line)

	if err != nil {
		fmt.Println(err)
	}

	// Debug data, ugly right ? :))
	if rt.IsDebugMode() {
		fmt.Println("\n### DEBUG ###")
		Print(rt, ast)
		fmt.Print("#############\n\n")
	}

	result, err := Eval(rt, ast)
	if err != nil {
		fmt.Printf("Error: %s\n", err)
		return
	}

	Print(rt, result)
}

/** TODO:
Replace the readline implementation with go-prompt.
*/

// REPL executes a Read Eval Print Loop locally reading from stdin and
// writing to stdout
func REPL(debug bool) {
	rt := MakeRuntime(debug)
	rt.CreateNS("user", "REPL", true)

	rl, err := readline.NewEx(&readline.Config{
		Prompt:            "> ",
		HistoryFile:       filepath.Join(os.Getenv("HOME"), ".serene.history"),
		InterruptPrompt:   "^C",
		EOFPrompt:         "exit",
		HistorySearchFold: true,
	})
	if err != nil {
		panic(err)
	}
	rl.HistoryEnable()
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
