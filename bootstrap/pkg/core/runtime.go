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
	"io/ioutil"
	"os"
	"path"
	"strings"
)

/** TODO:
Create an IRuntime interface to avoid using INamespace directly
*/

/** TODO:
Handle concurrency on the runtime level
*/
type loadedForms struct {
	source string
	forms  *Block
}

type Runtime struct {
	namespaces map[string]Namespace
	currentNS  string
	paths      []string
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

func (r *Runtime) setCurrentNS(nsName string) bool {
	_, ok := r.namespaces[nsName]

	if ok {
		r.currentNS = nsName
		return true
	}
	return false
}

func (r *Runtime) GetNS(ns string) (*Namespace, bool) {
	namespace, ok := r.namespaces[ns]
	return &namespace, ok
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

func nsNameToPath(ns string) string {
	replacer := strings.NewReplacer(
		".", "/",
		//"-", "_",
	)
	return replacer.Replace(ns)
}

func (r *Runtime) LoadNS(ns string) (*loadedForms, IError) {
	nsFile := nsNameToPath(ns)
	for _, loadPath := range r.paths {
		// TODO: Hardcoding the suffix??? ewwww, fix it.
		possibleFile := path.Join(loadPath, nsFile+".srn")

		_, err := os.Stat(possibleFile)

		if err != nil {
			continue
		}

		data, err := ioutil.ReadFile(possibleFile)

		if err != nil {
			readError := MakeError(
				r,
				fmt.Sprintf("error while reading the file at %s", possibleFile),
			)
			readError.WithError(err)
			return nil, readError
		}

		body, e := ReadString(string(data))
		if e != nil {
			return nil, e
		}

		return &loadedForms{possibleFile, body}, nil
	}

	// TODO: Add the load paths to the error message here
	return nil, MakeError(r, fmt.Sprintf("Can't find the namespace '%s' in any of load paths.", ns))
}

func (r *Runtime) RequireNS(ns string) (*Namespace, IError) {
	// TODO: use a hashing algorithm to avoid reloading an unchanged namespace
	loadedForms, err := r.LoadNS(ns)

	if err != nil {
		return nil, err
	}

	body := loadedForms.forms
	source := loadedForms.source

	if body.Count() == 0 {
		return nil, MakeError(
			r,
			fmt.Sprintf("The '%s' ns source code doesn't start with an 'ns' form.", ns),
		)
	}
	namespace := MakeNS(ns, source)
	namespace.setForms(body)

	return &namespace, nil
}

func (r *Runtime) InsertNS(nsName string, ns *Namespace) {
	r.namespaces[nsName] = *ns
}

func MakeRuntime(paths []string, debug bool) *Runtime {
	return &Runtime{
		namespaces: map[string]Namespace{},
		currentNS:  "",
		debugMode:  debug,
		paths:      paths,
	}
}
