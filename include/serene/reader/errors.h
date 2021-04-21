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

#ifndef SERENE_READER_ERRORS_H
#define SERENE_READER_ERRORS_H
#include "serene/errors.h"

namespace serene {
namespace reader {

class ReadError : public std::exception {
private:
  char *message;

public:
  ReadError(char *msg) : message(msg){};
  const char *what() const throw() { return message; }
};

class MissingFileError : public llvm::ErrorInfo<MissingFileError> {

  using llvm::ErrorInfo<MissingFileError>::log;
  using llvm::ErrorInfo<MissingFileError>::convertToErrorCode;

public:
  static char ID;
  std::string path;

  // TODO: Move this to an error namespace somewhere.
  int file_is_missing = int();

  void log(llvm::raw_ostream &os) const {
    os << "File does not exist: " << path << "\n";
  }

  MissingFileError(llvm::StringRef path) : path(path.str()){};
  std::error_code convertToErrorCode() const {
    return make_error_code(errc::no_such_file_or_directory);
  }
};

} // namespace reader
} // namespace serene
#endif
