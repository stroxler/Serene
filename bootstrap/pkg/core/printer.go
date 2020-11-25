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

func Print(rt *Runtime, ast IPrintable) {
	fmt.Println(ast.String())
}

func PrintError(rt *Runtime, err IError) {
	loc := err.GetLocation()
	fmt.Printf("Error: %s\nAt: %d to %d\n", err.String(), loc.GetStart(), loc.GetEnd())

}
