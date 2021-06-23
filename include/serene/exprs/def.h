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

#ifndef EXPRS_DEF_H
#define EXPRS_DEF_H

#include "serene/context.h"
#include "serene/errors/error.h"
#include "serene/exprs/expression.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>

namespace serene {

namespace exprs {
class List;

/// This data structure represent the Lisp symbol. Just a symbol
/// in the context of the AST and nothing else.
class Def : public Expression {

public:
  std::string binding;
  Node value;

  Def(reader::LocationRange &loc, llvm::StringRef binding, Node &v)
      : Expression(loc), binding(binding), value(v){};

  Def(Def &d) = delete;

  ExprType getType() const;
  std::string toString() const;
  MaybeNode analyze(SereneContext &);
  void generateIR(serene::Namespace &);

  static bool classof(const Expression *e);

  /// Create a Def node out a list. The list should contain the
  /// correct `def` form like `(def blah value)`. This function
  /// is supposed to be used in the semantic analysis phase.
  ///
  /// \param ctx The semantic analysis context object.
  /// \param list the list containing the `def` form
  static MaybeNode make(SereneContext &ctx, List *list);
  ~Def() = default;
};

} // namespace exprs
} // namespace serene

#endif
