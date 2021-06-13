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

#ifndef EXPRS_SYMBOL_H
#define EXPRS_SYMBOL_H

#include "serene/context.h"
#include "serene/exprs/expression.h"

#include "llvm/ADT/StringRef.h"

#include <string>

namespace serene {

namespace exprs {

/// This data structure represent the Lisp symbol. Just a symbol
/// in the context of the AST and nothing else.
class Symbol : public Expression {

public:
  std::string name;

  Symbol(reader::LocationRange &loc, llvm::StringRef name)
      : Expression(loc), name(name){};

  Symbol(Symbol &s) : Expression(s.location) { this->name = s.name; }

  ExprType getType() const;
  std::string toString() const;

  static bool classof(const Expression *e);

  MaybeNode analyze(SereneContext &);
  void generateIR(serene::Namespace &){};

  ~Symbol() = default;
};

} // namespace exprs
} // namespace serene

#endif
