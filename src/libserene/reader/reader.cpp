/* -*- C++ -*-
 * Serene Programming Language
 *
 * Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "serene/reader/reader.h"

#include "serene/errors/constants.h"
#include "serene/exprs/list.h"
#include "serene/exprs/number.h"
#include "serene/exprs/symbol.h"
#include "serene/namespace.h"

#include <assert.h>
#include <cctype>
#include <fstream>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SMLoc.h>
#include <memory>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LogicalResult.h>
#include <string>

namespace serene {

namespace reader {

Reader::Reader(SereneContext &ctx, llvm::StringRef buffer, llvm::StringRef ns,
               llvm::Optional<llvm::StringRef> filename)
    : ctx(ctx), ns(ns), filename(filename), buf(buffer),
      currentLocation(Location(ns, filename)) {
  READER_LOG("Setting the first char of the buffer");
  currentChar          = buf.begin() - 1;
  currentPos           = 1;
  currentLocation.line = 1;
  currentLocation.col  = 1;
};

Reader::Reader(SereneContext &ctx, llvm::MemoryBufferRef buffer,
               llvm::StringRef ns, llvm::Optional<llvm::StringRef> filename)
    : Reader(ctx, buffer.getBuffer(), ns, filename){};

Reader::~Reader() { READER_LOG("Destroying the reader"); }

void Reader::advanceByOne() {
  currentChar++;
  currentPos++;
  currentLocation.col++;

  if (*currentChar == '\n') {
    READER_LOG("Detected end of line");

    if (readEOL) {
      currentLocation.col = 1;
      currentLocation.line++;
    }

    readEOL = true;
  } else {
    if (readEOL) {
      currentLocation.line++;
      currentLocation.col = 1;
    }
    readEOL = false;
  }

  READER_LOG("Moving to Char: " << *currentChar << " at location: "
                                << currentLocation.toString());
};
void Reader::advance(bool skipWhitespace) {
  if (skipWhitespace) {
    for (;;) {
      auto next = currentChar + 1;

      if (!isspace(*next)) {
        return;
      }

      advanceByOne();
    }
  } else {
    advanceByOne();
  }
};

const char *Reader::nextChar(bool skipWhitespace, unsigned count) {
  if (!skipWhitespace) {
    READER_LOG("Next char: " << *(currentChar + count));
    return currentChar + count;
  }

  auto c = currentChar + 1;
  while (isspace(*c)) {
    c++;
  };

  READER_LOG("Next char: " << *c);
  return c;
};

bool Reader::isEndOfBuffer(const char *c) {
  return *c == '\0' || currentPos > buf.size() || *c == EOF;
};

Location Reader::getCurrentLocation() { return currentLocation.clone(); };

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

  auto c = nextChar();
  advance();

  LocationRange loc(getCurrentLocation());

  if (!isdigit(*c)) {
    ctx.diagEngine->emitSyntaxError(loc, errors::InvalidDigitForNumber);
    exit(1);
  }

  for (;;) {
    number += *c;
    c     = nextChar(false);
    empty = false;

    if (isdigit(*c) || *c == '.') {
      if (*c == '.' && floatNum == true) {
        loc = LocationRange(getCurrentLocation());
        ctx.diagEngine->emitSyntaxError(loc, errors::TwoFloatPoints);
        exit(1);
      }

      if (*c == '.') {
        floatNum = true;
      }

      advance();
      continue;
    }
    break;
  }

  if ((std::isalpha(*c) && !empty) || empty) {
    advance();
    loc.start = getCurrentLocation();
    ctx.diagEngine->emitSyntaxError(loc, errors::InvalidDigitForNumber);
    exit(1);
  }

  loc.end = getCurrentLocation();
  return exprs::make<exprs::Number>(loc, number, neg, floatNum);
};

/// Reads a symbol. If the symbol looks like a number
/// If reads it as number
exprs::Node Reader::readSymbol() {
  READER_LOG("Reading a symbol...");
  LocationRange loc;
  auto c = nextChar();

  if (!this->isValidForIdentifier(*c) || isEndOfBuffer(c) || isspace(*c)) {
    advance();
    loc = LocationRange(getCurrentLocation());
    std::string msg;

    if (*c == ')') {
      msg = "An extra ')' is detected.";
    }

    ctx.diagEngine->emitSyntaxError(loc, errors::InvalidCharacterForSymbol,
                                    msg);
    exit(1);
  }

  if (*c == '-') {
    auto next = nextChar(false, 2);
    if (isdigit(*next)) {
      // Swallow the -
      advance();
      return readNumber(true);
    }
  }

  if (isdigit(*c)) {
    return readNumber(false);
  }

  std::string sym("");
  advance();

  for (;;) {
    sym += *c;
    c = nextChar();

    if (!isEndOfBuffer(c) &&
        ((!(isspace(*c)) && this->isValidForIdentifier(*c)))) {
      advance();
      continue;
    }
    break;
  }

  loc.end = getCurrentLocation();
  return exprs::make<exprs::Symbol>(loc, sym);
};

/// Reads a list recursively
exprs::Node Reader::readList() {
  READER_LOG("Reading a list...");

  auto c = nextChar();
  advance();

  auto list = exprs::makeAndCast<exprs::List>(getCurrentLocation());

  // TODO: Replace the assert with an actual check.
  assert(*c == '(');

  bool list_terminated = false;

  do {
    auto c = nextChar(true);

    if (isEndOfBuffer(c)) {
      advance(true);
      advance();
      list->location.end = getCurrentLocation();
      ctx.diagEngine->emitSyntaxError(list->location,
                                      errors::EOFWhileScaningAList);
      exit(1);
    }

    switch (*c) {
    case ')':
      advance(true);
      advance();
      list_terminated    = true;
      list->location.end = getCurrentLocation();
      break;

    default:
      advance(true);
      list->append(readExpr());
    }

  } while (!list_terminated);

  return list;
};

/// Reads an expression by dispatching to the proper reader function.
exprs::Node Reader::readExpr() {
  auto c = nextChar(true);

  READER_LOG("Read char at `readExpr`: " << *c);

  if (isEndOfBuffer(c)) {
    return nullptr;
  }

  switch (*c) {
  case '(': {
    advance(true);
    return readList();
  }

  default:
    advance(true);
    return readSymbol();
  }
};

/// Reads all the expressions in the reader's buffer as an AST.
/// Each expression type (from the reader perspective) has a
/// reader function.
Result<exprs::Ast> Reader::read() {

  for (size_t current_pos = 0; current_pos < buf.size();) {
    auto c = nextChar(true);

    if (isEndOfBuffer(c)) {
      break;
    }

    advance(true);

    auto tmp{readExpr()};

    if (tmp) {
      this->ast.push_back(move(tmp));
    } else {
      break;
    }
  }

  return Result<exprs::Ast>::success(std::move(this->ast));
};

Result<exprs::Ast> read(SereneContext &ctx, const llvm::StringRef input,
                        llvm::StringRef ns,
                        llvm::Optional<llvm::StringRef> filename) {
  reader::Reader r(ctx, input, ns, filename);
  auto ast = r.read();
  return ast;
}

Result<exprs::Ast> read(SereneContext &ctx, const llvm::MemoryBufferRef input,
                        llvm::StringRef ns,
                        llvm::Optional<llvm::StringRef> filename) {
  reader::Reader r(ctx, input, ns, filename);

  auto ast = r.read();
  return ast;
}
} // namespace reader
} // namespace serene