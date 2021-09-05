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

#include "llvm/Support/ErrorHandling.h"

#include <assert.h>
#include <cctype>
#include <fstream>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SMLoc.h>
#include <memory>
#include <string>

namespace serene {

namespace reader {

Reader::Reader(SereneContext &ctx, llvm::StringRef buffer)
    : ctx(ctx), buf(buffer){};

Reader::Reader(SereneContext &ctx, llvm::MemoryBufferRef buffer)
    : ctx(ctx), buf(buffer.getBuffer()){};

Reader::~Reader() { READER_LOG("Destroying the reader"); }

const char *Reader::getChar(bool skip_whitespace) {
  for (;;) {
    if (current_char == NULL) {
      READER_LOG("Setting the first char of the buffer");
      current_char = buf.begin();
      current_pos  = 1;
    } else {
      current_char++;
      current_pos++;
    }

    READER_LOG("Current Char: " << *current_char);
    incLocation(current_location, current_char);

    if (skip_whitespace == true && isspace(*current_char)) {
      READER_LOG("Skip whitespace is true and the char is a whitespace");
      continue;
    } else {
      return current_char;
    }
  }
};

void Reader::ungetChar() {
  READER_LOG("Unread Char: " << *current_char);
  current_char--;
  current_pos--;
  // The char that we just unget
  decLocation(current_location, current_char);
};

bool Reader::isEndOfBuffer(const char *c) {
  return *c == '\0' || current_pos > buf.size() || *c == EOF;
};

Location Reader::getCurrentLocation() { return current_location.clone(); };

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

  if (std::isalnum(c)) {
    return true;
  }
  return false;
}

/// Reads a number,
/// \param neg whether to read a negative number or not.
exprs::Node Reader::readNumber(bool neg) {
  READER_LOG("Reading a number...");
  std::string number(neg ? "-" : "");
  bool floatNum = false;
  bool empty    = false;

  LocationRange loc;
  auto c = getChar(false);

  loc.start = getCurrentLocation();

  while (!isEndOfBuffer(c) &&
         ((!(isspace(*c)) && ((isdigit(*c)) | (*c == '.'))))) {

    if (*c == '.' && floatNum == true) {
      ctx.sourceManager.PrintMessage(
          llvm::errs(), llvm::SMLoc::getFromPointer(c),
          ctx.sourceManager.DK_Error,
          llvm::formatv("Two float points in a number?\n", c));
      exit(1);
    }

    if (*c == '.') {
      floatNum = true;
    }

    number += *c;
    c     = getChar(false);
    empty = false;
  }

  if (!empty) {
    ungetChar();
    loc.end = getCurrentLocation();
    return exprs::make<exprs::Number>(loc, number, neg, floatNum);
  }

  return nullptr;
};

/// Reads a symbol. If the symbol looks like a number
/// If reads it as number
exprs::Node Reader::readSymbol() {
  READER_LOG("Reading a symbol...");
  bool empty = true;
  auto c     = getChar(false);

  READER_LOG("Reading a symbol...");
  if (!this->isValidForIdentifier(*c)) {

    // TODO: Replece this with a tranceback function or something to raise
    // synatx error.
    ctx.sourceManager.PrintMessage(
        llvm::errs(), llvm::SMLoc::getFromPointer(c),
        ctx.sourceManager.DK_Error,
        llvm::formatv("Invalid character at the start of a symbol: '{0}'\n",
                      c));
    exit(1);
  }

  if (*c == '-') {
    auto next = getChar(false);
    ungetChar();
    if (isdigit(*next)) {
      return readNumber(true);
    }
  }

  if (isdigit(*c)) {
    ungetChar();
    return readNumber(false);
  }

  std::string sym("");
  LocationRange loc;
  loc.start = getCurrentLocation();

  while (!isEndOfBuffer(c) &&
         ((!(isspace(*c)) && this->isValidForIdentifier(*c)))) {
    sym += *c;
    c     = getChar(false);
    empty = false;
  }

  if (!empty) {
    ungetChar();
    loc.end = getCurrentLocation();
    return exprs::make<exprs::Symbol>(loc, sym);
  }

  llvm_unreachable("Unpredicted symbol read scenario");
  return nullptr;
};

/// Reads a list recursively
exprs::Node Reader::readList() {
  READER_LOG("Reading a list...");
  auto list = exprs::makeAndCast<exprs::List>(getCurrentLocation());

  auto c = getChar(true);

  // TODO: Replace the assert with an actual check.
  assert(*c == '(');

  bool list_terminated = false;

  do {
    auto c = getChar(true);

    if (isEndOfBuffer(c)) {
      ctx.sourceManager.PrintMessage(
          llvm::errs(), llvm::SMLoc::getFromPointer(c),
          ctx.sourceManager.DK_Error,
          llvm::formatv("EOF reached before closing of list"));

      exit(1);
    }

    switch (*c) {
    case ')':
      list_terminated    = true;
      list->location.end = getCurrentLocation();
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
  auto c = getChar(true);
  READER_LOG("Read char at `readExpr`: " << *c << " << " << current_pos << "|"
                                         << buf.size() << " BB "
                                         << isEndOfBuffer(c));
  ungetChar();

  if (isEndOfBuffer(c)) {
    return nullptr;
  }

  switch (*c) {
  case '(': {
    return readList();
  }
  default:
    return readSymbol();
  }
};

/// Reads all the expressions in the reader's buffer as an AST.
/// Each expression type (from the reader perspective) has a
/// reader function.
Result<exprs::Ast> Reader::read() {
  // auto c = getChar(true);

  // while (!isEndOfBuffer(c)) {
  for (size_t current_pos = 0; current_pos < buf.size();) {
    // ungetChar();
    auto tmp{readExpr()};

    if (tmp) {
      this->ast.push_back(move(tmp));
    } else {
      break;
    }
    // c = getChar(true);
  }

  return Result<exprs::Ast>::success(std::move(this->ast));
};

Result<exprs::Ast> read(SereneContext &ctx, const llvm::StringRef input) {
  reader::Reader r(ctx, input);
  auto ast = r.read();
  return ast;
}

Result<exprs::Ast> read(SereneContext &ctx, const llvm::MemoryBufferRef input) {
  reader::Reader r(ctx, input);
  auto ast = r.read();
  return ast;
}
} // namespace reader
} // namespace serene
