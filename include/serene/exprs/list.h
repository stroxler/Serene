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

#ifndef EXPRS_LIST_H
#define EXPRS_LIST_H

#include "serene/context.h"
#include "serene/exprs/expression.h"

#include "llvm/ADT/Optional.h"

#include <string>

namespace serene {

namespace exprs {

/// This class represents a List in the AST level and not the List as the data
/// type.
class List : public Expression {
public:
  // Internal elements of the lest (small vector of shared pointers to
  // expressions)
  Ast elements;

  List(const List &l);               // Copy ctor
  List(List &&e) noexcept = default; // Move ctor

  List(const reader::LocationRange &loc) : Expression(loc){};
  List(const reader::LocationRange &loc, Node &e);
  List(const reader::LocationRange &loc, Ast elems);

  ExprType getType() const;
  std::string toString() const;

  void append(Node);

  size_t count() const;

  Ast from(uint begin);
  // llvm::MutableArrayRef<Expression> from(uint begin);

  llvm::Optional<Expression *> at(uint index);

  std::vector<Node>::const_iterator cbegin();
  std::vector<Node>::const_iterator cend();
  std::vector<Node>::iterator begin();
  std::vector<Node>::iterator end();

  MaybeNode analyze(SereneContext &);
  void generateIR(serene::Namespace &){};

  static bool classof(const Expression *e);

  ~List() = default;
};

} // namespace exprs
} // namespace serene

#endif
