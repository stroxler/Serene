/* -*- C++ -*-
 * Serene programming language.
 *
 *  Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "serene/reader/reader.h"

#include "serene/exprs/list.h"
#include "serene/exprs/number.h"
#include "serene/exprs/symbol.h"
#include "serene/namespace.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"

#include <assert.h>
#include <fstream>
#include <memory>
#include <string>

namespace serene {

namespace reader {

Reader::Reader(const llvm::StringRef input) { this->setInput(input); };

/// Set the input of the reader.
///\param input Set the input to the given string
void Reader::setInput(const llvm::StringRef input) {
  current_location = Location::unit();
  ast.clear();
  input_stream.clear();
  input_stream.write(input.str().c_str(), input.size());
};

Reader::~Reader() { READER_LOG("Destroying the reader"); }

/// Return the next character in the buffer and moves the location.
///\param skip_whitespace If true it will skip whitespaces and EOL chars
/// \return next char in the buffer.
char Reader::getChar(bool skip_whitespace) {
  for (;;) {
    char c = input_stream.get();

    this->current_char = c;

    // TODO: Handle the end of line with respect to the OS.
    // increase the current position in the buffer with respect to the end
    // of line.
    inc_location(current_location, c == '\n');

    if (skip_whitespace == true && isspace(c)) {
      continue;
    } else {
      return c;
    }
  }
};

/// Moves back the location by one char. Basically unreads the last character.
void Reader::ungetChar() {
  input_stream.unget();
  // The char that we just unget
  dec_location(current_location, this->current_char == '\n');
};

/// A predicate function indicating whether the given char `c` is a valid
/// char for the starting point of a symbol or not.
bool Reader::isValidForIdentifier(char c) {
  switch (c) {
  case '!':
  case '$':
  case '%':
  case '&':
  case '*':
  case '+':
  case '-':
  case '.':
  case '~':
  case '/':
  case ':':
  case '<':
  case '=':
  case '>':
  case '?':
  case '@':
  case '^':
  case '_':
    return true;
  }

  if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
      (c >= '0' && c <= '9')) {
    return true;
  }
  return false;
}

/// Reads a number,
/// \param neg whether to read a negative number or not.
exprs::Node Reader::readNumber(bool neg) {
  std::string number(neg ? "-" : "");
  bool floatNum = false;
  bool empty    = false;

  LocationRange loc;
  char c = getChar(false);

  loc.start = current_location;

  while (c != EOF &&
         ((!(isspace(c)) && ((c >= '0' && c <= '9') | (c == '.'))))) {

    if (c == '.' && floatNum == true) {

      llvm::errs() << "Two float points in a number?\n";
      // TODO: Return a proper error
      return nullptr;
    }

    if (c == '.') {
      floatNum = true;
    }
    number += c;
    c     = getChar(false);
    empty = false;
  }

  if (!empty) {
    ungetChar();
    loc.end = current_location;
    return exprs::make<exprs::Number>(loc, number, neg, floatNum);
  }

  return nullptr;
};

/// Reads a symbol. If the symbol looks like a number
/// If reads it as number
exprs::Node Reader::readSymbol() {
  bool empty = true;
  char c     = getChar(false);

  READER_LOG("Reading symbol");
  if (!this->isValidForIdentifier(c)) {

    // TODO: Replece this with a tranceback function or something to raise
    // synatx error.
    llvm::errs() << llvm::formatv(
        "Invalid character at the start of a symbol: '{0}'\n", c);
    exit(1);
  }

  if (c == '-') {
    char next = getChar(false);
    if (next >= '0' && next <= '9') {
      ungetChar();
      return readNumber(true);
    }
  }

  if (c >= '0' && c <= '9') {
    ungetChar();
    return readNumber(false);
  }

  std::string sym("");
  LocationRange loc;
  loc.start = current_location;

  while (c != EOF && ((!(isspace(c)) && this->isValidForIdentifier(c)))) {
    sym += c;
    c     = getChar(false);
    empty = false;
  }

  if (!empty) {
    ungetChar();
    loc.end = current_location;
    return exprs::make<exprs::Symbol>(loc, sym);
  }

  // TODO: it should never happens
  return nullptr;
};

/// Reads a list recursively
exprs::Node Reader::readList() {
  auto list = exprs::makeAndCast<exprs::List>(current_location);

  char c = getChar(true);
  assert(c == '(');

  bool list_terminated = false;

  do {
    char c = getChar(true);

    switch (c) {
    case EOF:
      throw ReadError(const_cast<char *>("EOF reached before closing of list"));
    case ')':
      list_terminated    = true;
      list->location.end = current_location;

      break;

    default:
      ungetChar();
      list->append(readExpr());
    }

  } while (!list_terminated);

  return list;
};

/// Reads an expression by dispatching to the proper reader function.
exprs::Node Reader::readExpr() {
  char c = getChar(false);
  READER_LOG("CHAR: " << c);

  ungetChar();

  switch (c) {
  case '(': {

    return readList();
  }
  case EOF:
    return nullptr;

  default:
    return readSymbol();
  }
};

/// Reads all the expressions in the reader's buffer as an AST.
/// Each expression type (from the reader perspective) has a
/// reader function.
Result<exprs::Ast> Reader::read() {
  char c = getChar(true);

  while (c != EOF) {
    ungetChar();
    auto tmp{readExpr()};

    if (tmp) {
      this->ast.push_back(move(tmp));
    }

    c = getChar(true);
  }

  return Result<exprs::Ast>::success(std::move(this->ast));
};

/// Reads the input into an AST and prints it out as string again.
void Reader::toString() {
  auto maybeAst      = read();
  std::string result = "";

  if (!maybeAst) {
    throw std::move(maybeAst.getError());
  }

  exprs::Ast ast = std::move(maybeAst.getValue());

  for (auto &node : ast) {
    result = llvm::formatv("{0} {1}", result, node->toString());
  }
};

/// Reads all the expressions from the file provided via its path
// in the reader as an AST.
/// Each expression type (from the reader perspective) has a
/// reader function.
Result<exprs::Ast> FileReader::read() {

  // TODO: Add support for relative path as well
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(file);

  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    llvm::errs() << llvm::formatv("File: '{0}'\n", file);
    llvm::errs() << "Use absolute path for now\n";
    return Result<exprs::Ast>::error(llvm::make_error<MissingFileError>(file));
  }

  reader->setInput(fileOrErr.get()->getBuffer().str());
  return reader->read();
}

/// Reads the input into an AST and prints it out as string again.
void FileReader::toString() {
  auto maybeAst = this->read();
  exprs::Ast ast;

  if (!maybeAst) {
    throw std::move(maybeAst.getError());
  }

  ast = std::move(maybeAst.getValue());

  std::string result = "";
  for (auto &node : ast) {
    result = llvm::formatv("{0} {1}", result, node->toString());
  }
  llvm::outs() << result << "\n";
}

FileReader::~FileReader() {
  delete this->reader;
  READER_LOG("Destroying the file reader");
}

Result<exprs::Ast> read(llvm::StringRef input) {
  reader::Reader r(input);
  auto ast = r.read();
  return ast;
}
} // namespace reader
} // namespace serene
