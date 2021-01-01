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

// Package errors is dedicated to hold the error numbers and description of
// each knowen error in Serene
package errors

type Errno uint

const (
	E0000 Errno = iota // THE DESCRIPTION IS NOT SET
	E0001
	E0002
)

var ErrorsDescription map[Errno]string = map[Errno]string{
	E0000: `Can't find any description for this error.`,
	E0001: `
Namespaces are fundamental units in Serene. Each file has to start with
a namespace declaration with a name that matches the path of the file.

For example imagine haivng a file with the following path
'/home/user/xyz/src/example/abc.srn' and '/home/user/xyz/src' is
in the load path. The namespace path to the file would be 'example.abc' so
that file has to contain a 'ns' form as the first expression with
'example.abc' as the name just like:

(ns example.abc)
...rest of the file...

Since comments are not expressions it's ok to start a file by comments
followed by the 'ns' form`,

	E0002: `
Functions expect a certain number of argument. The number of arguments
that you're passing to the function doesn't match with it's signature.
To fix the problem double check the function signature and make sure
that you're passing the correct number of arguments to it`,
}
