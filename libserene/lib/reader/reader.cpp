/* -*- C++ -*-
 * Serene Programming Language
 *
 * Copyright (c) 2019-2022 Sameer Rahmani <lxsameer@gnu.org>
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

#include "serene/errors.h"
#include "serene/exprs/expression.h"
#include "serene/exprs/list.h"
#include "serene/exprs/number.h"
#include "serene/exprs/symbol.h"
#include "serene/namespace.h"
#include "serene/utils.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SMLoc.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LogicalResult.h>

#include <assert.h>
#include <cctype>
#include <fstream>
#include <memory>
#include <string>

namespace serene {

namespace reader {
// LocationRange::LocationRange(const LocationRange &loc) {
//   start = loc.start.clone();
//   end   = loc.end.clone();
// }

/// Return the string represenation of the location.
std::string Location::toString() const {
  return llvm::formatv("{0}:{1}", line, col);
};

Location Location::clone() const {
  return Location{ns, filename, c, line, col, knownLocation};
}

mlir::Location Location::toMLIRLocation(SereneContext &ctx) {
  // TODO: Create a new Location attribute that is namespace base
  if (filename.hasValue()) {
    return mlir::FileLineColLoc::get(&ctx.mlirContext, filename.getValue(),
                                     line, col);
  }
  return mlir::FileLineColLoc::get(&ctx.mlirContext, ns, line, col);
}
/// Increase the given location by one and set the line/col value in respect to
/// the `newline` in place.
/// \param loc The `Location` data
/// \param c A pointer to the current char that the location has to point to
void incLocation(Location &loc, const char *c) {
  // TODO: Handle the end of line with respect to the OS.
  // increase the current position in the buffer with respect to the end
  // of line.
  auto newline = *c == '\n';

  if (!newline) {
    loc.col++;
  } else {
    loc.line++;
    loc.col = 0;
  }
}

/// decrease the given location by one and set the line/col value in respect to
/// the `newline` in place.
/// \param loc The `Location` data
/// \param c A pointer to the current char that the location has to point to
void decLocation(Location &loc, const char *c) {
  // TODO: Handle the end of line with respect to the OS.
  // increase the current position in the buffer with respect to the end
  // of line.
  auto newline = *c == '\n';

  if (newline) {
    loc.line = loc.line == 0 ? 0 : loc.line - 1;

    // We don't move back the `col` value because we simply don't know it
  } else {
    loc.col = loc.col == 0 ? 0 : loc.col - 1;
  }
}

Reader::Reader(SereneContext &ctx, llvm::StringRef buffer, llvm::StringRef ns,
               llvm::Optional<llvm::StringRef> filename)
    : ctx(ctx), ns(ns), filename(filename), buf(buffer),
      currentLocation(Location(ns, filename)) {
  UNUSED(this->ctx);
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
      const auto *next = currentChar + 1;

      if (isspace(*next) == 0) {
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

  const auto *c = currentChar + 1;
  while (isspace(*c) != 0) {
    c++;
  };

  READER_LOG("Next char: " << *c);
  return c;
};

bool Reader::isEndOfBuffer(const unsigned char *c) {
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

  return std::isalnum(c) != 0;
}

/// Reads a number,
/// \param neg whether to read a negative number or not.
exprs::MaybeNode Reader::readNumber(bool neg) {
  READER_LOG("Reading a number...");
  std::string number(neg ? "-" : "");
  bool floatNum = false;
  bool empty    = false;

  const auto *c = nextChar();
  advance();

  LocationRange loc(getCurrentLocation());

  if (isdigit(*c) == 0) {
    return errors::makeError(ctx, errors::InvalidDigitForNumber, loc);
  }

  for (;;) {
    number += *c;
    c     = nextChar(false);
    empty = false;

    if ((isdigit(*c) != 0) || *c == '.') {
      if (*c == '.' && floatNum) {
        loc = LocationRange(getCurrentLocation());
        return errors::makeError(ctx, errors::TwoFloatPoints, loc);
      }

      if (*c == '.') {
        floatNum = true;
      }

      advance();
      continue;
    }
    break;
  }

  if (((std::isalpha(*c) != 0) && !empty) || empty) {
    advance();
    loc.start = getCurrentLocation();
    return errors::makeError(ctx, errors::InvalidDigitForNumber, loc);
  }

  loc.end = getCurrentLocation();
  return exprs::make<exprs::Number>(loc, number, neg, floatNum);
};

/// Reads a symbol. If the symbol looks like a number
/// If reads it as number
exprs::MaybeNode Reader::readSymbol() {
  READER_LOG("Reading a symbol...");
  LocationRange loc;
  const auto *c = nextChar();

  if (!this->isValidForIdentifier(*c) || isEndOfBuffer(c) ||
      (isspace(*c) != 0)) {
    advance();
    loc = LocationRange(getCurrentLocation());
    std::string msg;

    if (*c == ')') {
      msg = "An extra ')' is detected.";
    }

    return errors::makeError(ctx, errors::InvalidCharacterForSymbol, loc, msg);
  }

  if (*c == '-') {
    const auto *next = nextChar(false, 2);
    if (isdigit(*next) != 0) {
      // Swallow the -
      advance();
      return readNumber(true);
    }
  }

  if (isdigit(*c) != 0) {
    return readNumber(false);
  }

  std::string sym;
  advance();

  for (;;) {
    sym += *c;
    c = nextChar();

    if (!isEndOfBuffer(c) &&
        ((((isspace(*c)) == 0) && this->isValidForIdentifier(*c)))) {
      advance();
      continue;
    }
    break;
  }

  // TODO: Make sure that the symbol has 0 or 1 '/'.

  // TODO: Make sure that `/` is not at the start or at the end of the symbol

  loc.end = getCurrentLocation();
  return exprs::makeSuccessfulNode<exprs::Symbol>(loc, sym, this->ns);
};

/// Reads a list recursively
exprs::MaybeNode Reader::readList() {
  READER_LOG("Reading a list...");

  const auto *c = nextChar();
  advance();

  auto list = exprs::makeAndCast<exprs::List>(getCurrentLocation());

  // TODO: Replace the assert with an actual check.
  assert(*c == '(');

  bool list_terminated = false;

  do {
    const auto *c = nextChar(true);

    if (isEndOfBuffer(c)) {
      advance(true);
      advance();
      list->location.end = getCurrentLocation();
      return errors::makeError(ctx, errors::EOFWhileScaningAList,
                               list->location);
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
      auto expr = readExpr();
      if (!expr) {
        return expr;
      }

      list->append(*expr);
    }

  } while (!list_terminated);

  return list;
};

/// Reads an expression by dispatching to the proper reader function.
exprs::MaybeNode Reader::readExpr() {
  const auto *c = nextChar(true);

  READER_LOG("Read char at `readExpr`: " << *c);

  if (isEndOfBuffer(c)) {
    return exprs::EmptyNode;
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
exprs::MaybeAst Reader::read() {

  for (size_t current_pos = 0; current_pos < buf.size();) {
    const auto *c = nextChar(true);

    if (isEndOfBuffer(c)) {
      break;
    }

    advance(true);

    auto tmp = readExpr();

    if (tmp) {
      if (*tmp == nullptr) {
        break;
      }

      this->ast.push_back(std::move(*tmp));

    } else {
      return tmp.takeError();
    }
  }

  return std::move(this->ast);
};

exprs::MaybeAst read(SereneContext &ctx, const llvm::StringRef input,
                     llvm::StringRef ns,
                     llvm::Optional<llvm::StringRef> filename) {
  reader::Reader r(ctx, input, ns, filename);
  auto ast = r.read();
  return ast;
}

exprs::MaybeAst read(SereneContext &ctx, const llvm::MemoryBufferRef input,
                     llvm::StringRef ns,
                     llvm::Optional<llvm::StringRef> filename) {
  reader::Reader r(ctx, input, ns, filename);

  auto ast = r.read();
  return ast;
}
} // namespace reader
} // namespace serene
