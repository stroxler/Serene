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

// Package dl provides the necessary interface to interact with share
// libraries
package dl

/*
#cgo linux LDFLAGS: -ldl
#cgo pkg-config: libffi
#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>
#include <stdint.h>

#include <stdio.h>

static uintptr_t openShareLib(const char* path, char** err) {
	void* h = dlopen(path, RTLD_NOW|RTLD_GLOBAL);
	if (h == NULL) {
		*err = (char*)dlerror();
	}
	return (uintptr_t)h;
}

static void* shareLibLookup(uintptr_t h, const char* name, char** err) {
	void* r = dlsym((void*)h, name);
	if (r == NULL) {
		*err = (char*)dlerror();
	}
	return r;
}
*/

// import "C"
// import (
// 	"fmt"
// 	"unsafe"
// )

// type SharedLib struct{}

// func Open(libPath string) (*SharedLib, error) {
// 	cPath := make([]byte, C.PATH_MAX+1)
// 	cRelName := make([]byte, len(libPath)+1)
// 	copy(cRelName, libPath)

// 	// If the given libPath exists, fill the cPath with the absolute path
// 	// to the file (it follows symlinks).
// 	if C.realpath(
// 		(*C.char)(unsafe.Pointer(&cRelName[0])),
// 		(*C.char)(unsafe.Pointer(&cPath[0]))) == nil {
// 		return nil, fmt.Errorf("can't find the shared library '%s'", libPath)
// 	}

// 	var cErr *C.char
// 	h := C.openShareLib((*C.char)(unsafe.Pointer(&cPath[0])), &cErr)

// 	if h == 0 {
// 		// lock.Unlock()
// 		return nil, fmt.Errorf("failed to open shared library: '%s' due to: '%s'", libPath, C.GoString(cErr))
// 	}

// 	initTask := C.shareLibLookup(h, C.CString("bar"), &cErr)

// 	fmt.Printf(">> %p \n", initTask)

// 	return &SharedLib{}, nil
// }
