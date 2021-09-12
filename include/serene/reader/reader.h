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

  /// When we're dealing with the end of line we need to know what is the col
  /// number for that EOL
  unsigned prevCol = 0;

  bool readEOL = false;

  /// Returns a clone of the current location
  Location getCurrentLocation();
  /// Returns the next character from the stream.
  /// @param skip_whitespace An indicator to whether skip white space like chars
  /// or not
  const char *getChar(bool skipWhitespace);

  const char *nextChar();

  /// Unreads the current character by moving the char pointer to the previous
  /// char.
  void ungetChar();

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
