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
	"strings"

	"github.com/chzyer/readline"
	"serene-lang.org/bootstrap/pkg/ast"
)

func rep(rt *Runtime, line string) {
	ast, err := ReadString("*REPL*", line)

	if err != nil {
		PrintError(rt, err)
		return
	}

	// Debug data, ugly right ? :))
	if rt.IsDebugMode() {
		fmt.Printf("[DEBUG] Parsed AST: %s\n", ast.String())
	}

	result, e := Eval(rt, ast)
	if e != nil {
		PrintError(rt, e)
		return
	}
	Prn(rt, result)
}

/** TODO:
Replace the readline implementation with go-prompt.
*/

// REPL executes a Read Eval Print Loop locally reading from stdin and
// writing to stdout
func REPL(flags map[string]bool) {
	cwd, err := os.Getwd()
	if err != nil {
		panic(err)
	}

	rt := MakeRuntime([]string{cwd}, flags)

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

	fmt.Println(`
 _______ _______ ______ _______ _______ _______
|     __|    ___|   __ \    ___|    |  |    ___|
|__     |    ___|      <    ___|       |    ___|
|_______|_______|___|__|_______|__|____|_______|


Serene's bootstrap interpreter is used to
bootstrap the Serene's compiler.

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

func Run(flags map[string]bool, args []string) {
	cwd, e := os.Getwd()
	if e != nil {
		panic(e)
	}

	rt := MakeRuntime([]string{cwd}, flags)

	if len(args) == 0 {

		PrintError(rt, MakePlainError("'run' command needs at least one argument"))
		os.Exit(1)
	}

	ns := args[0]
	nsAsBuffer := strings.Split(ns, "")
	source := &ast.Source{Buffer: &nsAsBuffer, Path: "*input-argument*"}
	node := MakeNode(source, 0, len(ns))
	nsSym, err := MakeSymbol(node, ns)

	if err != nil {
		PrintError(rt, err)
		os.Exit(1)
	}

	loadedNS, err := requireNS(rt, nsSym)

	if err != nil {
		PrintError(rt, err)
		os.Exit(1)
	}

	rt.InsertNS(ns, loadedNS)
	inserted := rt.setCurrentNS(loadedNS.GetName())

	if !inserted {
		err := MakeError(
			rt,
			loadedNS,
			fmt.Sprintf(
				"the namespace '%s' didn't get inserted in the runtime.",
				loadedNS.GetName()),
		)
		PrintError(rt, err)
		os.Exit(1)
	}

	// Evaluating the body of the loaded ns (Check for ns validation happens here)
	loadedNS, err = EvalNSBody(rt, loadedNS)
	if err != nil {
		PrintError(rt, err)
		os.Exit(1)
	}

	mainBinding := loadedNS.GetRootScope().Lookup(rt, "main")

	if mainBinding == nil {
		PrintError(rt, MakePlainError(fmt.Sprintf("can't find the 'main' function in '%s' namespace", ns)))
		os.Exit(1)
	}

	if mainBinding.Value.GetType() != ast.Fn {
		PrintError(rt, MakePlainError("'main' is not a function"))
		os.Exit(1)
	}

	mainFn := mainBinding.Value.(*Function)

	var fnArgs []IExpr
	var argNode Node

	if len(args) > 1 {
		for _, arg := range args[1:] {
			node := MakeNodeFromExpr(mainFn)
			fnArgs = append(fnArgs, MakeString(node, arg))
		}
		argNode = MakeNodeFromExprs(fnArgs)
	} else {
		argNode = node
	}

	_, err = mainFn.Apply(rt, loadedNS.GetRootScope(), mainFn.Node, MakeList(argNode, fnArgs))

	if err != nil {
		PrintError(rt, err)
		os.Exit(1)
	}
}
