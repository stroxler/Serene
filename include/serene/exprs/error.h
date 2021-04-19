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

#ifndef EXPRS_ERROR_H
#define EXPRS_ERROR_H

#include "serene/exprs/expression.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace serene {

namespace exprs {

/// This enum represent the expression type and **not** the value type.
enum class ErrType {
  Syntax,
  Semantic,
  Compile,
};

/// This data structure represent the Lisp error. This type of expression
/// doesn't show up in the AST but the compiler might rewrite the AST
/// to contains error expressions
class Error : public Expression {

public:
  ErrType errorType;
  node target;
  std::string message;

  Error(ErrType err, node &t, llvm::StringRef msg)
      : Expression(t->location), errorType(err), target(t), message(msg){};

  // Error(reader::LocationRange &loc, ErrType err, node &t, llvm::StringRef
  // msg)
  //   : Expression(loc), errorType(err), target(t),  message(msg) {};

  ExprType getType() const;
  std::string toString() const;

  static bool classof(const Expression *e);

  maybe_node analyze(reader::SemanticContext &);

  ~Error() = default;
};

} // namespace exprs
} // namespace serene

#endif
