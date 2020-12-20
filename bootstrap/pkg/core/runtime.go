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

	"serene-lang.org/bootstrap/pkg/dl"
)

/** TODO:
Create an IRuntime interface to avoid using INamespace directly
*/

/** TODO:
Handle concurrency on the runtime level
*/

// loadedForms is used as "pair" implementation to keep the loaded
// expressions and the source where the expressions are coming from
type loadedForms struct {
	source string
	forms  *Block
}

// TODO: Make the Runtime and it's fields thread safe

// Runtime is the most important data structure in Serene which hold
// the necessary information and the state of the interpreter at runtime (duh!).
// At any given time and thread only on Runtime has to exist and we always need
// to pass a pointer to the runtime around and avoid copying. (We don't have
// multithread support just yet but the Runtime must be thread safe).
type Runtime struct {
	// A mapping from ns names (e.g some.ns.over.there) to the namespace
	// data. This hashmap is owner of the namespaces, meaning that we only
	// pass pointers to the namespaces around and any mutation has to happen
	// here
	namespaces map[string]Namespace

	// A mapping from the builtin function names to the corresponding
	// NativeFunction struct that implements them as expressions (IExpr).
	// native functions are those which can be special form as well but
	// they are more suited to be a function and at the same time we
	// can't implement them in Serene itself.
	builtins map[string]NativeFunction

	// currentNS is the fully qualified name of the current namespace which
	// is being processed (evaluates) at any given time. Since it's not
	// thread safe at the moment we need to be careful changeing its value.
	currentNS string

	// paths is an array of filesystem paths that have we need to look into
	// in order to find and load the namespaces. Similar to `load_path` in other
	// languages
	paths []string

	// A to turn on the verbose mode, FOR DEVELOPMENT USE ONLY
	debugMode bool
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

// GetNS returns a pointer to the `Namespace` specified with the given name `ns`
// on the runtime.
func (r *Runtime) GetNS(ns string) (*Namespace, bool) {
	namespace, ok := r.namespaces[ns]
	return &namespace, ok
}

// CreateNS is a helper function to create a namespace and set it to be
// the current namespace of the runtime. `MakeNS` is much preferred
func (r *Runtime) CreateNS(name string, source string, setAsCurrent bool) {
	ns := MakeNS(name, source)

	if setAsCurrent {
		r.currentNS = name
	}
	r.namespaces[name] = ns
}

// IsQQSimplificationEnabled returns a boolean value indicating whether
// simplification of quasiquotation is enabled or not. If yes, we have
// to replace the quasiquote expanded forms with a simplier form to gain
// a better performance.
func (r *Runtime) IsQQSimplificationEnabled() bool {
	// TODO: read the value of this flag from the arguments of serene
	//       and set the default to true
	return false
}

// nsNameToPath converts a namespace name to the filesystem equivilant path
func nsNameToPath(ns string) string {
	replacer := strings.NewReplacer(
		".", "/",
		//"-", "_",
	)
	return replacer.Replace(ns) + "srn"
}

// LoadNS looks up the namespace specified by the given name `ns`
// and reads the content as expressions (parse it) and returns the
// expressions.
func (r *Runtime) LoadNS(ns string) (*loadedForms, IError) {
	nsFile := nsNameToPath(ns)
	for _, loadPath := range r.paths {
		possibleFile := path.Join(loadPath, nsFile)

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

func (r *Runtime) InsertNS(nsName string, ns *Namespace) {
	r.namespaces[nsName] = *ns
}

func (r *Runtime) LookupBuiltin(k string) IExpr {
	builtinfn, ok := r.builtins[k]

	if ok {
		return &builtinfn
	}

	return nil
}

// MakeRuntime creates a Runtime and returns a pointer to it. Any
// runtime initialization such as adding default namespaces and vice
// versa has to happen here.
func MakeRuntime(paths []string, debug bool) *Runtime {
	_, e := dl.Open("/home/lxsameer/src/serene/serene/bootstrap/examples/ffi/foo/libfoo.so")
	if e != nil {
		panic(e)
	}
	rt := Runtime{
		namespaces: map[string]Namespace{},
		currentNS:  "",
		debugMode:  debug,
		paths:      paths,
	}

	rt.builtins = BUILTINS
	return &rt
}
