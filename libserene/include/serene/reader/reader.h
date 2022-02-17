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

/**
 * Commentary:
 * `Reader` is the base parser class and accepts a buffer like objenct (usually
 * `llvm::StringRef`) as the input and parsess it to create an AST (look at the
 * `serene::exprs::Expression` class).
 *
 * The parsing algorithm is quite simple and it is a LL(2). It means that, we
 * start parsing the input from the very first character and parse the input
 * one char at a time till we reach the end of the input. Please note that
 * when we call the `advance` function to move forward in the buffer, we
 * can't go back. In order to look ahead in the buffer without moving in the
 * buffer we use the `nextChar` method.
 *
 * We have dedicated methods to read different forms like `list`, `symbol`
 * `number` and etc. Each of them return a `MaybeNode` that in the success
 * case contains the node and an `Error` on the failure case.
 */

#ifndef SERENE_READER_READER_H
#define SERENE_READER_READER_H

#include "serene/errors.h"
#include "serene/exprs/expression.h"
#include "serene/exprs/list.h"
#include "serene/exprs/symbol.h"
#include "serene/reader/location.h"
#include "serene/serene.h"

#include <system_error>

#include <llvm/Support/Debug.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Support/raw_ostream.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define READER_LOG(...)                  \
  DEBUG_WITH_TYPE("READER", llvm::dbgs() \
                                << "[READER]: " << __VA_ARGS__ << "\n");

namespace serene::reader {

/// Base reader class which reads from a string directly.
class Reader {
private:
  SereneContext &ctx;

  llvm::StringRef ns;
  llvm::Optional<llvm::StringRef> filename;

  const char *currentChar = NULL;

  llvm::StringRef buf;

  /// The position tracker that we will use to determine the end of the
  /// buffer since the buffer might not be null terminated
  size_t currentPos = -1;

  Location currentLocation;

  bool readEOL = false;

  /// Returns a clone of the current location
  Location getCurrentLocation();
  /// Returns the next character from the stream.
  /// @param skip_whitespace An indicator to whether skip white space like chars
  /// or not
  void advance(bool skipWhitespace = false);
  void advanceByOne();

  const char *nextChar(bool skipWhitespace = false, unsigned count = 1);

  /// Returns a boolean indicating whether the given input character is valid
  /// for an identifier or not.
  static bool isValidForIdentifier(char c);

  // The property to store the ast tree
  exprs::Ast ast;

  exprs::MaybeNode readSymbol();
  exprs::MaybeNode readNumber(bool);
  exprs::MaybeNode readList();
  exprs::MaybeNode readExpr();

  bool isEndOfBuffer(const char *);

public:
  Reader(SereneContext &ctx, llvm::StringRef buf, llvm::StringRef ns,
         llvm::Optional<llvm::StringRef> filename);
  Reader(SereneContext &ctx, llvm::MemoryBufferRef buf, llvm::StringRef ns,
         llvm::Optional<llvm::StringRef> filename);

  // void setInput(const llvm::StringRef string);

  /// Parses the the input and creates a possible AST out of it or errors
  /// otherwise.
  exprs::MaybeAst read();

  ~Reader();
};

/// Parses the given `input` string and returns a `Result<ast>`
/// which may contains an AST or an `llvm::Error`
exprs::MaybeAst read(SereneContext &ctx, llvm::StringRef input,
                     llvm::StringRef ns,
                     llvm::Optional<llvm::StringRef> filename);
exprs::MaybeAst read(SereneContext &ctx, llvm::MemoryBufferRef input,
                     llvm::StringRef ns,
                     llvm::Optional<llvm::StringRef> filename);
} // namespace serene::reader

#endif
