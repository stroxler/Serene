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

#include "serene/reader/location.h"
#include <memory>

namespace serene {

/// Contains all the builtin expressions including those which do not appear in
/// the syntax directly. Like function definitions.
namespace exprs {

/// This enum represent the expression type and **not** the value type.
enum class ExprType {
  Symbol,
  List,
};

/// An abstract class which locatable expressions should inherit from
class Locatable {
public:
  Locatable(reader::LocationRange loc) : location(loc){};
  reader::LocationRange location;
};

/// The polymorphic type that works as the entry point to the exprs system.
/// Each expression has to define the interface of the `ExpressionConcept`
/// class as generic functions. **REMEMBER TO NOT INHERIT FROM THESE CLASSES**
class Expression {
public:
  /// Creates a new expression by moving the given object of the type T into
  /// a new internal container.
  ///
  /// \param e and expression of type T
  template <typename T> Expression(T e) : self(new Impl<T>(std::move(e))){};

  /// The copy constructor which actually just move the other expression into
  /// a new implementation container.
  ///
  /// \param e is the other expression to copy from
  Expression(const Expression &e) : self(e.self->copy_()){}; // Copy ctor
  Expression(Expression &&e) noexcept = default;             // Move ctor

  Expression &operator=(const Expression &e);
  Expression &operator=(Expression &&e) noexcept = default;

  /// Returns the type of the expression. More precisely, It returns the type
  /// of the expression that it contains.
  ///
  /// \return The type of expression.
  ExprType getType();

  /// Return the string representation of the expression in the context
  /// of the AST. Think of it as dump of the AST for each expression.
  ///
  /// \return the exoression in string format.
  std::string toString();

  /// Create a new Expression of type `T` and forwards any given parameter
  /// to the constructor of type `T`. This is the **official way** to create
  /// a new `Expression`. Here is an example:
  /// \code
  /// auto list = Expression::make<List>();
  /// \endcode
  ///
  /// \param loc A `serene::reader::LocationRange` instance to point to exact
  /// location of the expression in the input string.
  /// \param[args] Any argument with any type passed to this function will be
  ///              passed to the constructor of type T.
  /// \return A new expression containing a value of type T and act as tyep T.
  template <typename T, typename... Args>
  static Expression make(Args &&...args) {
    return Expression(T::build(std::forward<Args>(args)...));
  };

  // template <typename T> static Expression make(reader::LocationRange &&loc) {
  //   Expression e(T(std::forward<reader::LocationRange>(loc)));
  //   return e;
  // };

  /// The generic interface which each type of expression has to implement
  /// in order to act like an `Expression`
  class ExpressionConcept {
  public:
    virtual ~ExpressionConcept() = default;
    virtual ExpressionConcept *copy_() const = 0;

    /// Return the type of the expression
    virtual ExprType getType() = 0;

    /// Return the string representation of the expression in the context
    /// of the AST. Think of it as dump of the AST for each expression
    virtual std::string toString() = 0;
  };

  /// The generic implementation of `ExpressionConcept` which acts as the
  /// dispatcher on type.
  template <typename T> struct Impl : ExpressionConcept {
    Impl(T e) : expr(std::move(e)){};

    ExpressionConcept *copy_() const { return new Impl(*this); }

    /// In order to make llvm's RTTI to work we need this method.
    ExprType getType() { return expr.getType(); }

    std::string toString() { return expr.toString(); }

    T expr;
  };

  /// The internal container to keep the object implementing the
  /// `ExpressionConcept`. This might be a `List` for example or a `Symbol`.
  std::unique_ptr<ExpressionConcept> self;
};

} // namespace exprs
} // namespace serene

#endif
