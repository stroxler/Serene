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

#ifndef EXPRS_EXPRESSION_H
#define EXPRS_EXPRESSION_H

#include "serene/context.h"
#include "serene/reader/location.h"
#include "serene/utils.h"
#include <memory>

namespace serene {

namespace reader {
class SereneContext;
}

/// Contains all the builtin AST expressions including those which do not appear
/// in the syntax directly. Like function definitions.
namespace exprs {

/// This enum represent the expression type and **not** the value type.
enum class ExprType {
  Symbol,
  List,
  Number,
  Def,
  Error,
  Fn,
  Call,
};

/// The string represantion of built in expr types (NOT DATATYPES).
static const char *exprTypes[] = {
    "Symbol", "List", "Number", "Def", "Error", "Fn", "Call",
};

class Expression;

using Node = std::shared_ptr<Expression>;
using MaybeNode = Result<Node>;

using Ast = std::vector<Node>;
using MaybeAst = Result<Ast>;

/// The base class of the expressions which provides the common interface for
/// the expressions to implement.
class Expression {
public:
  /// The location range provide information regarding to where in the input
  /// string the current expression is used.
  reader::LocationRange location;

  Expression(const reader::LocationRange &loc) : location(loc){};
  virtual ~Expression() = default;

  /// Returns the type of the expression. We need this funciton to perform
  /// dynamic casting of expression object to implementations such as lisp or
  /// symbol.
  virtual ExprType getType() const = 0;

  /// The AST representation of an expression
  virtual std::string toString() const = 0;

  /// Analyzes the semantics of current node and return a new node in case
  /// that we need to semantically rewrite the current node and replace it with
  /// another node. For example to change from a List containing `(def a b)`
  /// to a `Def` node that represents defining a new binding.
  ///
  /// \param ctx is the context object of the semantic analyzer.
  virtual MaybeNode analyze(SereneContext &ctx) = 0;
};

/// Create a new `node` of type `T` and forwards any given parameter
/// to the constructor of type `T`. This is the **official way** to create
/// a new `Expression`. Here is an example:
/// \code
/// auto list = make<List>();
/// \endcode
///
/// \param[args] Any argument with any type passed to this function will be
///              passed to the constructor of type T.
/// \return A unique pointer to an Expression
template <typename T, typename... Args> Node make(Args &&...args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
};

/// Create a new `node` of type `T` and forwards any given parameter
/// to the constructor of type `T`. This is the **official way** to create
/// a new `Expression`. Here is an example:
/// \code
/// auto list = makeAndCast<List>();
/// \endcode
///
/// \param[args] Any argument with any type passed to this function will be
///              passed to the constructor of type T.
/// \return A unique pointer to a value of type T.
template <typename T, typename... Args>
std::shared_ptr<T> makeAndCast(Args &&...args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
};

/// Convert the given AST to string by calling the `toString` method
/// of each node.
std::string astToString(const Ast *);
/// Converts the given ExprType to string.
std::string stringifyExprType(ExprType);

/// Converts the given AST to string and prints it out
void dump(Ast &);

} // namespace exprs
} // namespace serene

#endif
