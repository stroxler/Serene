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

#ifndef EXPRS_TRAITS_H
#define EXPRS_TRAITS_H

#include "serene/context.h"
#include "serene/reader/location.h"
#include "serene/reader/traits.h"
#include "serene/traits.h"
#include "serene/utils.h"

namespace serene::exprs {
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

template <typename ConcreteType>
class ITypeable : public TraitBase<ConcreteType, ITypeable> {
public:
  ITypeable(){};
  ITypeable(const ITypeable &) = delete;
  ExprType getType() const { return this->Object().getType(); }
};

template <typename ConcreteType>
class IAnalyzable : public TraitBase<ConcreteType, IAnalyzable> {
public:
  IAnalyzable(){};
  IAnalyzable(const IAnalyzable &) = delete;
  auto analyze(SereneContext &ctx);
};

template <typename ConcreteType>
class SExp : public WithTrait<ConcreteType, ITypeable,
                              serene::reader::ILocatable, serene::IDebuggable> {

protected:
  serene::reader::LocationRange location;
  SExp(const serene::reader::LocationRange &loc) : location(loc){};

public:
  serene::reader::LocationRange where() const { return this->location; }
};

}; // namespace serene::exprs

#endif
