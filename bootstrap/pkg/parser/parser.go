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

// Package parser provides necessary functions to generate an AST
// from an input
package parser

import (
	"errors"
	"fmt"
	"strings"
	"unicode"

	"serene-lang.org/bootstrap/pkg/types"
)

var validChars = []rune{'!', '$', '%', '&', '*', '+', '-', '.', '~', '/', ':', '<', '=', '>', '?', '@', '^', '_'}

type StringParser struct {
	buffer []string
	pos    int
}

// Implementing IParsable for StringParser ---
func (sp *StringParser) next(skipWhitespace bool) *string {
	if sp.pos >= len(sp.buffer) {
		return nil
	}
	char := sp.buffer[sp.pos]
	sp.pos = sp.pos + 1

	if skipWhitespace && char == " " {
		return sp.next(skipWhitespace)
	}

	return &char
}

func (sp *StringParser) peek(skipWhitespace bool) *string {
	if sp.pos >= len(sp.buffer) {
		return nil
	}

	c := sp.buffer[sp.pos]
	if c == " " && skipWhitespace {
		sp.pos = sp.pos + 1
		return sp.peek(skipWhitespace)
	}
	return &c
}

func (sp *StringParser) back() {
	if sp.pos > 0 {
		sp.pos = sp.pos - 1
	}
}

func (sp *StringParser) GetLocation() int {
	return sp.pos
}

// END: IParsable ---
func contains(s []rune, c rune) bool {
	for _, v := range s {
		if v == c {
			return true
		}
	}

	return false
}

func isValidForSymbol(char string) bool {
	c := rune(char[0])
	return contains(validChars, c) || unicode.IsLetter(c) || unicode.IsDigit(c)
}

func readRawSymbol(parser IParsable) (types.IExpr, error) {
	c := parser.peek(false)
	var symbol string

	if c == nil {
		return nil, errors.New("Unexpected EOF while parsing a symbol")
	}

	if isValidForSymbol(*c) {
		symbol = *c
	} else {

		return nil, fmt.Errorf("Unexpected character: got '%s', expected a symbol at %s",
			*c,
			parser.GetLocation(),
		)
	}

	for {
		c := parser.next(false)

		if c == nil {
			break
		}

		if isValidForSymbol(*c) {
			symbol = symbol + *c
		} else {
			parser.back()
			break
		}
	}

	// TODO: Add support for ns qualified symbols
	return types.MakeSymbol(symbol), nil
}

func readNumber(parser IParsable, neg bool) (types.IExpr, error) {
	isDouble := false
	result := ""

	if neg {
		result = "-"
	}

	for {
		c := parser.next(false)

		if c == nil {
			break
		}

		if *c == "." && isDouble {
			fmt.Println(result)
			return nil, errors.New("a double with more that one '.' ???")
		}

		if *c == "." {
			isDouble = true
			result = result + *c
			continue
		}

		// Weird, But go won't stop complaining without this swap
		char := *c
		r := rune(char[0])
		if unicode.IsDigit(r) {
			result = result + *c
		} else {
			parser.back()
			break
		}
	}
	fmt.Println(result)
	return types.MakeNumberFromStr(result, isDouble)
}

func readSymbol(parser IParsable) (types.IExpr, error) {
	c := parser.peek(false)

	if c == nil {
		return nil, errors.New("unexpected end of file while scanning a symbol")
	}

	// if c == "\"" {
	// 	return readString(parser)
	// }

	// Weird, But go won't stop complaining without this swap
	char := *c
	r := rune(char[0])
	if unicode.IsDigit(r) {
		return readNumber(parser, false)
	}

	if *c == "-" {
		parser.next(true)
		c := parser.peek(false)

		// Weird, But go won't stop complaining without this swap
		char := *c
		r := rune(char[0])

		if unicode.IsDigit(r) {
			return readNumber(parser, true)
		} else {
			// Unread '-'
			parser.back()
			return readRawSymbol(parser)
		}

	}
	return readRawSymbol(parser)
}

func readList(parser IParsable) (types.IExpr, error) {
	list := []types.IExpr{}

	for {
		c := parser.peek(true)
		if c == nil {
			return nil, errors.New("reaching the end of file while reading a list")
		}
		if *c == ")" {
			parser.next(true)
			break
		} else {
			val, err := readExpr(parser)

			if err != nil {
				return nil, err
			}
			list = append(list, val)

		}
	}

	return types.MakeList(list), nil
}

func readComment(parser IParsable) (types.IExpr, error) {
	for {
		c := parser.next(false)
		if c == nil || *c == "\n" {
			return nil, nil
		}
	}
}

func readQuotedExpr(parser IParsable) (types.IExpr, error) {
	expr, err := readExpr(parser)
	if err != nil {
		return nil, err
	}

	return types.MakeList([]types.IExpr{
		types.MakeSymbol("quote"),
		expr,
	}), nil
}

func readUnquotedExpr(parser IParsable) (types.IExpr, error) {
	c := parser.peek(true)

	if c == nil {
		return nil, errors.New("end of file while reading an unquoted expression")
	}

	var sym types.IExpr
	var err error
	var expr types.IExpr

	if *c == "@" {
		parser.next(true)
		sym = types.MakeSymbol("unquote-splicing")
		expr, err = readExpr(parser)

	} else {
		sym = types.MakeSymbol("unquote")
		expr, err = readExpr(parser)
	}

	if err != nil {
		return nil, err
	}

	return types.MakeList([]types.IExpr{sym, expr}), nil
}

func readQuasiquotedExpr(parser IParsable) (types.IExpr, error) {
	expr, err := readExpr(parser)
	if err != nil {
		return nil, err
	}

	return types.MakeList([]types.IExpr{
		types.MakeSymbol("quasiquote"),
		expr,
	}), nil
}

func readExpr(parser IParsable) (types.IExpr, error) {

loop:
	c := parser.next(true)

	if c == nil {
		// We're done reading
		return nil, nil
	}

	if *c == "'" {
		return readQuotedExpr(parser)
	}

	if *c == "~" {
		return readUnquotedExpr(parser)
	}

	if *c == "`" {
		return readQuasiquotedExpr(parser)
	}
	if *c == "(" {
		return readList(parser)
	}
	if *c == ";" {
		readComment(parser)
		goto loop
	}
	// case '[':
	// 	readVector(parser)

	// case '{':
	// 	readMap(parser)

	parser.back()
	return readSymbol(parser)

}

func ParseToAST(input string) (types.ASTree, error) {

	var ast types.ASTree
	parser := StringParser{
		buffer: strings.Split(input, ""),
		pos:    0,
	}

	for {
		expr, err := readExpr(&parser)
		if err != nil {
			return nil, err
		}

		if expr == nil {
			break
		}

		ast = append(ast, expr)
	}

	return ast, nil
}
