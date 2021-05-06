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

#ifndef SERENE_ERRORS_ERROR_H
#define SERENE_ERRORS_ERROR_H

#include "serene/context.h"
#include "serene/errors/constants.h"
#include "serene/errors/traits.h"
#include "serene/reader/location.h"
#include "serene/reader/traits.h"
#include "serene/traits.h"
//#include "serene/exprs/expression.h"
#include "llvm/Support/Error.h"

namespace serene::errors {

/// This enum represent the expression type and **not** the value type.
enum class ErrType {
  Syntax,
  Semantic,
  Compile,
};

/// This data structure represent the Lisp error. This type of expression
/// doesn't show up in the AST but the compiler might rewrite the AST
/// to contains error expressions
class Error
    : public WithTrait<Error, IError, reader::ILocatable, serene::IDebuggable> {
  reader::LocationRange location;
  ErrorVariant *variant;
  std::string message;

public:
  Error(reader::LocationRange &loc, ErrorVariant *err, llvm::StringRef msg)
      : location(loc), variant(err), message(msg){};

  std::string toString() const;
  reader::LocationRange &where();
  ErrorVariant *getVariant();
  std::string getMessage();
  ~Error() = default;
};

}; // namespace serene::errors

#endif
