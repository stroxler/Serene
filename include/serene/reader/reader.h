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
 * buffer we use the `nextChar` method-

 */

#ifndef READER_H
#define READER_H

#include "serene/errors.h"
#include "serene/exprs/expression.h"
#include "serene/exprs/list.h"
#include "serene/exprs/symbol.h"
#include "serene/reader/errors.h"
#include "serene/reader/location.h"
#include "serene/serene.h"

#include <llvm/Support/Debug.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
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
  bool isValidForIdentifier(char c);

  // The property to store the ast tree
  exprs::Ast ast;

  exprs::Node readSymbol();
  exprs::Node readNumber(bool);
  exprs::Node readList();
  exprs::Node readExpr();

  bool isEndOfBuffer(const char *);

public:
  Reader(SereneContext &ctx, llvm::StringRef buf, llvm::StringRef ns,
         llvm::Optional<llvm::StringRef> filename);
  Reader(SereneContext &ctx, llvm::MemoryBufferRef buf, llvm::StringRef ns,
         llvm::Optional<llvm::StringRef> filename);

  // void setInput(const llvm::StringRef string);

  /// Parses the the input and creates a possible AST out of it or errors
  /// otherwise.
  Result<exprs::Ast> read();

  ~Reader();
};

/// Parses the given `input` string and returns a `Result<ast>`
/// which may contains an AST or an `llvm::Error`
Result<exprs::Ast> read(SereneContext &ctx, const llvm::StringRef input,
                        llvm::StringRef ns,
                        llvm::Optional<llvm::StringRef> filename);
Result<exprs::Ast> read(SereneContext &ctx, const llvm::MemoryBufferRef but,
                        llvm::StringRef ns,
                        llvm::Optional<llvm::StringRef> filename);
} // namespace serene::reader
#endif
