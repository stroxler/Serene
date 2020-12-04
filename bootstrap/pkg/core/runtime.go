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
)

/** TODO:
Create an IRuntime interface to avoid using INamespace directly
*/

/** TODO:
Handle concurrency on the runtime level
*/

type Runtime struct {
	namespaces map[string]Namespace
	currentNS  string
	debugMode  bool
}

func (r *Runtime) IsDebugMode() bool {
	return r.debugMode
}

func (r *Runtime) CurrentNS() *Namespace {
	if r.currentNS == "" {
		panic("current ns is not set on the runtime.")
	}

	ns, ok := r.namespaces[r.currentNS]

	if !ok {
		panic(fmt.Sprintf("namespace '%s' doesn't exist in the runtime.", r.currentNS))
	}

	return &ns
}

func (r *Runtime) CreateNS(name string, source string, setAsCurrent bool) {
	ns := MakeNS(name, source)

	if setAsCurrent {
		r.currentNS = name
	}
	r.namespaces[name] = ns
}

func (r *Runtime) IsQQSimplificationEnabled() bool {
	// TODO: read the value of this flag from the arguments of serene
	//       and set the default to true
	return false
}

func MakeRuntime(debug bool) *Runtime {
	return &Runtime{
		namespaces: map[string]Namespace{},
		currentNS:  "",
		debugMode:  debug,
	}
}
