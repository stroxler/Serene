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

#include <memory>

namespace serene {

/// Contains all the builtin expressions including those which do not appear in
/// the syntax directly. Like function definitions.
namespace exprs {

/// The polymorphic type that works as the entry point to the exprs system.
/// Each expression has to define the interface of the `ExpressionConcept`
/// class as generic functions. **REMEMBER TO NOT INHERIT FROM THESE CLASSES**
class Expression {
public:
  template <typename T>
  Expression(T e) : self(new ExpressionImpl<T>(std::move(e))){};

private:
  /// The generic interface which each type of expression has to implement
  /// in order to act like an `Expression`
  struct ExpressionConcept {
    virtual ~ExpressionConcept() = default;
    virtual ExpressionConcept *copy_() const = 0;
  };

  /// The generic implementation of `ExpressionConcept` which acts as the
  /// dispatcher on type.
  template <typename T> struct ExpressionImpl : ExpressionConcept {
    ExpressionImpl(T e) : expr(std::move(e)){};
    ExpressionConcept *copy_() const { return new ExpressionImpl(*this); }

    T expr;
  };

  /// The internal container to keep the object implementing the
  /// `ExpressionConcept`. This might be a `List` for example or a `Symbol`.
  std::unique_ptr<ExpressionConcept> self;
};

} // namespace exprs
} // namespace serene

#endif
