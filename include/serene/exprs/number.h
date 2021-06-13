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

#ifndef EXPRS_NUMBER_H
#define EXPRS_NUMBER_H

#include "serene/context.h"
#include "serene/exprs/expression.h"
#include "serene/namespace.h"

#include "llvm/Support/FormatVariadic.h"

namespace serene {

namespace exprs {

/// This data structure represent a number. I handles float points, integers,
/// positive and negative numbers. This is not a data type representative.
/// So it won't cast to actual numeric types and it has a string container
/// to hold the parsed value.
struct Number : public Expression {
  // TODO: Use a variant here instead
  std::string value;

  bool isNeg;
  bool isFloat;

  Number(reader::LocationRange &loc, const std::string &num, bool isNeg,
         bool isFloat)
      : Expression(loc), value(num), isNeg(isNeg), isFloat(isFloat){};

  ExprType getType() const;
  std::string toString() const;
  MaybeNode analyze(SereneContext &ctx);

  static bool classof(const Expression *e);

  // TODO: This is horrible, we need to fix it after the mvp
  int toI64();

  void generateIR(serene::Namespace &);
  ~Number() = default;
};

} // namespace exprs
} // namespace serene

#endif
