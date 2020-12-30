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

// Parser Implementation:
// * `ParseToAST` is the entry point of the parser
// * It's a manual parser with look ahead factor of (1)
// * It parsers the input string to a tree of `IEpxr`s
//
// TODOs:
// * Add a shortcut for anonymous functions similar to `#(...)` clojure
//   syntax
// * Add the support for strings
// * Add the support for kewords
// * Add a shortcut for the `deref` function like `@x` => `(deref x)`

import (
	"strings"
	"unicode"

	"serene-lang.org/bootstrap/pkg/ast"
)

// An array of the valid characters that be be used in a symbol
var validChars = []rune{'!', '$', '%', '&', '*', '+', '-', '.', '~', '/', ':', '<', '=', '>', '?', '@', '^', '_'}

// IParsable defines the common interface which any parser has to implement.
type IParsable interface {
	// Reads the next character in the buffer with respect to skipWhitespace
	// parameter which basically jumps over whitespace and some conceptual
	// equivilant of a whitespace like '\n'
	next(skipWhitespace bool) *string

	// Similar to the `next` but it won't change the position in the buffer
	// so an imidiate `next` function after a `peek` will read the same char
	// but will move the position, and a series of `peek` calls will read the
	// same function over and over again without changing the position in the
	// buffer.
	peek(skipWhitespace bool) *string

	// Moves back the position by one in the buffer.
	back()

	// Returns the current position in the buffer
	GetLocation() int
	GetSource() *ast.Source
	Buffer() *[]string
}

// StringParser is an implementation of the  IParsable that operates on strings.
// To put it simply it parses input strings
type StringParser struct {
	buffer    []string
	pos       int
	source    string
	lineIndex []int
}

// Implementing IParsable for StringParser ---

// Returns the next character in the buffer
func (sp *StringParser) next(skipWhitespace bool) *string {
	if sp.pos >= len(sp.buffer) {
		return nil
	}
	char := sp.buffer[sp.pos]

	if char == "\n" {
		// Including the \n itself
		sp.lineIndex = append(sp.lineIndex, sp.pos+1)
	}

	sp.pos = sp.pos + 1

	if skipWhitespace && isSeparator(&char) {
		return sp.next(skipWhitespace)
	}

	return &char
}

// isSeparator returns a boolean indicating whether the given character `c`
// contains a separator or not. In a Lisp whitespace and someother characters
// are conceptually the same and we need to treat them the same as well.
func isSeparator(c *string) bool {

	if c == nil {
		return false
	}

	r := []rune(*c)[0]
	if r == ' ' || r == '\t' || r == '\n' || r == '\f' {
		return true
	}

	return false

}

// Return the character of the buffer without consuming it
func (sp *StringParser) peek(skipWhitespace bool) *string {
	if sp.pos >= len(sp.buffer) {
		return nil
	}

	c := sp.buffer[sp.pos]
	if isSeparator(&c) && skipWhitespace {
		sp.pos = sp.pos + 1
		return sp.peek(skipWhitespace)
	}
	return &c
}

// Move the char pointer back by one character
func (sp *StringParser) back() {
	if sp.pos > 0 {
		sp.pos = sp.pos - 1
	}
}

func (sp *StringParser) GetLocation() int {
	return sp.pos
}

func (sp *StringParser) GetSource() *ast.Source {
	return &ast.Source{
		Buffer:    &sp.buffer,
		Path:      sp.source,
		LineIndex: &sp.lineIndex,
	}
}

func (sp *StringParser) Buffer() *[]string {
	return &sp.buffer
}

// END: IParsable ---

// makeErrorAtPoint is a helper function which generates an `IError` that
// points at the current position of the buffer.
func makeErrorAtPoint(p IParsable, msg string, a ...interface{}) IError {
	n := MakeSinglePointNode(p.GetSource(), p.GetLocation())
	return MakeParsetimeErrorf(n, msg, a...)
}

// makeErrorFromError is a function which wraps a Golang error in an IError
func makeErrorFromError(parser IParsable, e error) IError {
	return makeErrorAtPoint(parser, "%w", e)
}

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

func readKeyword(parser IParsable) (IExpr, IError) {
	symbol, err := readRawSymbol(parser)
	if err != nil {
		return nil, err
	}

	node := MakeNodeFromExpr(symbol)
	return MakeKeyword(node, ":"+symbol.(*Symbol).String())
}

//readRawSymbol reads a symbol from the current position forward
func readRawSymbol(parser IParsable) (IExpr, IError) {
	c := parser.peek(false)
	var symbol string

	if c == nil {
		return nil, makeErrorAtPoint(parser, "unexpected enf of file while parsing a symbol")
	}

	// Does the symbol starts with a valid character or not
	if isValidForSymbol(*c) {
		parser.next(false)
		symbol = *c
	} else {
		return nil, makeErrorAtPoint(parser,
			"unexpected character: got '%s', expected a symbol at %d",
			*c,
			parser.GetLocation(),
		)
	}

	// read the rest of the symbol
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

	node := MakeNode(parser.GetSource(), parser.GetLocation()-len(symbol), parser.GetLocation())
	sym, err := MakeSymbol(node, symbol)

	if err != nil {
		err.SetNode(&node)
		return nil, err
	}

	return sym, nil
}

func readString(parser IParsable) (IExpr, IError) {
	str := ""

	for {
		c := parser.next(false)
		if c == nil {
			return nil, makeErrorAtPoint(parser, "reached end of file while scanning a string")
		}

		if *c == "\"" {
			node := MakeNode(parser.GetSource(), parser.GetLocation()-len(str), parser.GetLocation())
			return MakeString(node, str), nil
		}

		if *c == "\\" {
			c = parser.next(false)
			switch *c {
			case "n":
				str = str + "\n"
			case "t":
				str = str + "\t"
			case "r":
				str = str + "\r"
			case "\\":
				str = str + "\\"
			case "\"":
				str = str + "\""
			default:
				return nil, makeErrorAtPoint(parser, "Unsupported escape character: \\%s", *c)
			}
		} else {
			str = str + *c
		}
	}
}

// readNumber reads a number with respect to its sign and whether it's, a ...interface{}
// a decimal or a float
func readNumber(parser IParsable, neg bool) (IExpr, IError) {
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
			return nil, makeErrorAtPoint(parser, "a double with more that one '.' ???")
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

	value, err := MakeNumberFromStr(result, isDouble)

	if err != nil {
		return nil, makeErrorFromError(parser, err)
	}

	return value, nil
}

// readSymbol reads a symbol and return the appropriate type of expression
// based on the symbol conditions. For example it will read a number if the
// symbol starts with a number or a neg sign or a string if it starts with '\"'
// and a raw symbol otherwise
func readSymbol(parser IParsable) (IExpr, IError) {
	c := parser.peek(false)

	if c == nil {
		return nil, makeErrorAtPoint(parser, "unexpected end of file while scanning a symbol")
	}

	if *c == "\"" {
		parser.next(false)
		return readString(parser)
	}

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

// readList reads a List recursively.
func readList(parser IParsable) (IExpr, IError) {
	list := []IExpr{}

	for {
		c := parser.peek(true)
		if c == nil {
			return nil, makeErrorAtPoint(parser, "reaching the end of file while reading a list")
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

	node := MakeNodeFromExprs(list)
	node.location.DecStart(1)
	node.location.IncEnd(1)
	return MakeList(node, list), nil
}

func readComment(parser IParsable) (IExpr, IError) {
	for {
		c := parser.next(false)
		if c == nil || *c == "\n" {
			return nil, nil
		}
	}
}

// readQuotedExpr reads quoted expression ( lie 'something ) by replaceing the
// quote with a call to `quote` special form so 'something => (quote something)
func readQuotedExpr(parser IParsable) (IExpr, IError) {
	expr, err := readExpr(parser)
	if err != nil {
		return nil, err
	}

	symNode := MakeNode(parser.GetSource(), parser.GetLocation(), parser.GetLocation())
	sym, err := MakeSymbol(symNode, "quote")

	if err != nil {
		err.SetNode(&symNode)
		return nil, err
	}

	listElems := []IExpr{
		sym,
		expr,
	}

	listNode := MakeNodeFromExprs(listElems)
	listNode.location.DecStart(1)
	listNode.location.IncStart(1)
	return MakeList(listNode, listElems), nil
}

// readUnquotedExpr reads different unquoting expressions from their short representaions.
// ~a => (unquote a)
// ~@a => (unquote-splicing a)
// Note: `unquote` and `unquote-splicing` are not global functions or special, they are bounded
// to quasiquoted experssions only.
func readUnquotedExpr(parser IParsable) (IExpr, IError) {
	c := parser.peek(true)

	if c == nil {
		return nil, makeErrorAtPoint(parser, "end of file while reading an unquoted expression")
	}

	var sym IExpr
	var err IError
	var expr IExpr

	node := MakeNode(parser.GetSource(), parser.GetLocation(), parser.GetLocation())

	if *c == "@" {
		parser.next(true)
		sym, err = MakeSymbol(node, "unquote-splicing")
		if err != nil {
			err.SetNode(&node)
		} else {
			expr, err = readExpr(parser)
		}

	} else {
		sym, err = MakeSymbol(node, "unquote")
		if err != nil {
			err.SetNode(&node)
		} else {
			expr, err = readExpr(parser)
		}
	}

	if err != nil {
		return nil, err
	}

	listElems := []IExpr{sym, expr}
	listNode := MakeNodeFromExprs(listElems)
	listNode.location.DecStart(1)
	listNode.location.IncStart(1)
	return MakeList(listNode, listElems), nil
}

// readQuasiquotedExpr reads the backquote and replace it with a call
// to the `quasiquote` macro.
func readQuasiquotedExpr(parser IParsable) (IExpr, IError) {
	expr, err := readExpr(parser)
	if err != nil {
		return nil, err
	}

	node := MakeNode(parser.GetSource(), parser.GetLocation(), parser.GetLocation())
	sym, err := MakeSymbol(node, "quasiquote")
	if err != nil {
		err.SetNode(&node)
		return nil, err
	}

	listElems := []IExpr{sym, expr}
	listNode := MakeNodeFromExprs(listElems)
	listNode.location.DecStart(1)
	listNode.location.IncStart(1)

	return MakeList(listNode, listElems), nil
}

// readExpr reads one expression from the input. This function is the most
// important function in the parser which dispatches the call to different
// reader functions based on the first character
func readExpr(parser IParsable) (IExpr, IError) {

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

	if *c == ":" {
		return readKeyword(parser)
	}
	// if *c == "[" {
	// 	readVector(parser)
	// }

	// if *c == "{" {
	// 	readMap(parser)
	// }
	parser.back()
	return readSymbol(parser)

}

//ParseToAST is the entry function to the reader/parser which
// converts the `input` string to a `Block` of code. A block
// by itself is not something available to the language. It's
// just anbstraction for a ordered collection of expressions.
// It doesn't have anything to do with the concept of blocks
// from other programming languages.
func ParseToAST(source string, input string) (*Block, IError) {

	var ast Block
	parser := StringParser{
		buffer: strings.Split(input, ""),
		pos:    0,
		source: source,
	}

	for {
		expr, err := readExpr(&parser)
		if err != nil {
			return nil, err
		}

		if expr == nil {
			break
		}

		ast.Append(expr)
	}

	return &ast, nil
}
