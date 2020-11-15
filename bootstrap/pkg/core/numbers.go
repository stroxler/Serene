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
	"strconv"

	"serene-lang.org/bootstrap/pkg/ast"
)

type INumber interface {
	IExpr
	// Int() int
	// I8() int8
	// I16() int16
	// I32() int32
	I64() int64

	// UInt() uint
	// UI8() uint8
	// UI16() uint16
	// UI32() uint32
	// UI64() uint64

	//F32() float32
	F64() float64

	// Add(n Number) Number
	// TDOD: Add basic operators here
}

/** WARNING:
These are really stupid implementations of numbers we
need better implmentations later, but it's ok for something
to begin with
*/

type Integer struct {
	Node
	value int64
}

func (i Integer) Eval() IExpr {
	return &i
}

func (i Integer) GetType() ast.NodeType {
	return ast.Number
}

func (i Integer) String() string {
	return fmt.Sprintf("%d", i.value)
}

func (i Integer) ToDebugStr() string {
	return fmt.Sprintf("%#v", i)
}

func (i Integer) I64() int64 {
	return i.value
}

func (i Integer) F64() float64 {
	return float64(i.value)
}

type Double struct {
	Node
	value float64
}

func (d Double) Eval() IExpr {
	return &d
}

func (d Double) GetType() ast.NodeType {
	return ast.Number
}

func (d Double) String() string {
	return fmt.Sprintf("%f", d.value)
}

func (d Double) ToDebugStr() string {
	return fmt.Sprintf("%#v", d)
}

func (d Double) I64() int64 {
	return int64(d.value)
}

func (d Double) F64() float64 {
	return d.value
}

func MakeNumberFromStr(strValue string, isDouble bool) (INumber, error) {
	var ret INumber

	if isDouble {
		v, err := strconv.ParseFloat(strValue, 64)

		if err != nil {
			return nil, err
		}

		ret = Double{
			value: v,
		}
	} else {
		v, err := strconv.ParseInt(strValue, 10, 64)
		if err != nil {
			return nil, err
		}

		ret = Integer{
			value: v,
		}
	}

	return ret, nil
}
