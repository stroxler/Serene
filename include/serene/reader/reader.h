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

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#define READER_LOG(...) \
  DEBUG_WITH_TYPE("READER", llvm::dbgs() << __VA_ARGS__ << "\n");

namespace serene {
namespace reader {

/// Base reader class which reads from a string directly.
class Reader {
private:
  char current_char = ';'; // Some arbitary char to begin with
  std::stringstream input_stream;
  Location current_location{0, 0, 0};

  char getChar(bool skip_whitespace);
  void ungetChar();
  bool isValidForIdentifier(char c);

  // The property to store the ast tree
  exprs::Ast ast;
  exprs::Node readSymbol();
  exprs::Node readNumber(bool);
  exprs::Node readList();
  exprs::Node readExpr();

public:
  Reader() : input_stream(""){};
  Reader(const llvm::StringRef string);

  void setInput(const llvm::StringRef string);

  Result<exprs::Ast> read();

  // Dumps the AST data to stdout
  void toString();

  ~Reader();
};

/// A reader to read the content of a file as AST
class FileReader {
  std::string file;
  Reader *reader;

public:
  FileReader(const std::string file_name)
      : file(file_name), reader(new Reader()) {}
  // Dumps the AST data to stdout
  void toString();

  Result<exprs::Ast> read();

  ~FileReader();
};

/// Parses the given `input` string and returns a `Result<ast>`
/// which may contains an AST or an `llvm::Error`
Result<exprs::Ast> read(llvm::StringRef input);

} // namespace reader
} // namespace serene
#endif
