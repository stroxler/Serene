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
	"encoding/binary"
	"fmt"
	"math"
	"strconv"

	"serene-lang.org/bootstrap/pkg/ast"
	"serene-lang.org/bootstrap/pkg/hash"
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
	// ExecutionScope checkout IScopable
	scope IScope

	ExecutionScope
	value int64
}

func (i Integer) GetType() ast.NodeType {
	return ast.Number
}

func (i Integer) Hash() uint32 {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, uint64(i.value))
	return hash.Of(b)
}

func (i Integer) GetExecutionScope() IScope {
	return i.scope

}

func (i Integer) SetExecutionScope(scope IScope) {
	i.scope = scope
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

func MakeInteger(x interface{}) (*Integer, IError) {
	var value int64
	switch x.(type) {
	case uint32:
		value = int64(x.(uint32))
	case int32:
		value = int64(x.(int32))
	default:
		return nil, MakePlainError(fmt.Sprintf("don't know how to make 'integer' out of '%s'", x))
	}

	return &Integer{
		value: value,
	}, nil
}

type Double struct {
	Node
	// ExecutionScope checkout IScopable
	scope IScope
	value float64
}

func (d Double) GetType() ast.NodeType {
	return ast.Number
}

func (d Double) Hash() uint32 {
	b := make([]byte, 8)

	binary.BigEndian.PutUint64(b, math.Float64bits(d.value))
	return hash.Of(b)
}

func (d Double) String() string {
	return fmt.Sprintf("%f", d.value)
}

func (d Double) ToDebugStr() string {
	return fmt.Sprintf("%#v", d)
}

func (d Double) GetExecutionScope() IScope {
	return d.scope

}

func (d Double) SetExecutionScope(scope IScope) {
	d.scope = scope
}

func (d Double) I64() int64 {
	return int64(d.value)
}

func (d Double) F64() float64 {
	return d.value
}

func MakeNumberFromStr(n Node, strValue string, isDouble bool) (INumber, error) {
	var ret INumber

	if isDouble {
		v, err := strconv.ParseFloat(strValue, 64)

		if err != nil {
			return nil, err
		}

		ret = Double{
			Node:  n,
			value: v,
		}
	} else {
		v, err := strconv.ParseInt(strValue, 10, 64)
		if err != nil {
			return nil, err
		}

		ret = Integer{
			Node:  n,
			value: v,
		}
	}

	return ret, nil
}

func MakeDouble(x interface{}) (*Double, IError) {
	var value float64
	switch x.(type) {
	case uint32:
		value = float64(x.(uint32))
	case int32:
		value = float64(x.(int32))
	case float32:
		value = float64(x.(float32))
	default:
		return nil, MakePlainError(fmt.Sprintf("don't know how to make 'double' out of '%s'", x))
	}

	return &Double{
		value: value,
	}, nil
}
