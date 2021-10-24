/* -*- C++ -*-
 * Serene Programming Language
 *
 * Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SERENE_EXPRS_EXPRESSION_H
#define SERENE_EXPRS_EXPRESSION_H

#include "serene/context.h"
#include "serene/errors/error.h"
#include "serene/exprs/traits.h"
#include "serene/reader/location.h"
#include "serene/utils.h"

#include <mlir/IR/BuiltinOps.h>

#include <memory>

namespace serene {

/// Contains all the builtin AST expressions including those which do not appear
/// in the syntax directly. Like function definitions.
namespace exprs {

class Expression;

using Node     = std::shared_ptr<Expression>;
using ErrorPtr = std::shared_ptr<errors::Error>;

// tree? Yupe, Errors can be stackable which makes a vector of them a tree
using ErrorTree = std::vector<ErrorPtr>;

using MaybeNode = Result<Node, ErrorTree>;

using Ast      = std::vector<Node>;
using MaybeAst = Result<Ast, ErrorTree>;

static auto EmptyNode = MaybeNode::success(nullptr);

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

  /// Genenates the correspondig SLIR of the expressoin and attach it to the
  /// given module.
  ///
  /// \param ns The namespace that current expression is in it.
  /// \param m  The target MLIR moduleOp to attach the operations to
  virtual void generateIR(serene::Namespace &ns, mlir::ModuleOp &m) = 0;
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
template <typename T, typename... Args>
Node make(Args &&...args) {
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

/// The helper function to create a new `Node` and use that as the success case
// of a `Result`. It should be useds where every we want to return a `MaybeNode`
/// successfully
template <typename T, typename... Args>
Result<Node, ErrorTree> makeSuccessfulNode(Args &&...args) {
  return Result<Node, ErrorTree>::success(make<T>(std::forward<Args>(args)...));
};

/// The hlper function to create an Errorful `Result<T,...>` (`T` would be
/// either `Node` or `Ast` most of the time) with just one error created from
/// passing any argument to this function to the `serene::errors::Error`
/// constructor.
template <typename T, typename... Args>
Result<T, ErrorTree> makeErrorful(Args &&...args) {
  std::vector<ErrorPtr> v{
      std::move(makeAndCast<errors::Error>(std::forward<Args>(args)...))};
  return Result<T, ErrorTree>::error(v);
};

/// The hlper function to create an Error node (The failure case of a MaybeNod)
/// with just one error created from passing any argument to this function to
/// the `serene::errors::Error` constructor.
template <typename... Args>
MaybeNode makeErrorNode(Args &&...args) {
  std::vector<ErrorPtr> v{
      std::move(makeAndCast<errors::Error>(std::forward<Args>(args)...))};
  return MaybeNode::error(v);
};

/// Convert the given AST to string by calling the `toString` method
/// of each node.
SERENE_EXPORT std::string astToString(const Ast *);
/// Converts the given ExprType to string.
std::string stringifyExprType(ExprType);

/// Converts the given AST to string and prints it out
void dump(Ast &);

} // namespace exprs
} // namespace serene

#endif
