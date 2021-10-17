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

#ifndef EXPRS_NUMBER_H
#define EXPRS_NUMBER_H

#include "serene/context.h"
#include "serene/exprs/expression.h"
#include "serene/namespace.h"

#include <llvm/Support/FormatVariadic.h>

namespace serene {
namespace exprs {

/// This data structure represent a number. I handles float points, integers,
/// positive and negative numbers. This is not a data type representative.
/// So it won't cast to actual numeric types and it has a string container
/// to hold the parsed value.
struct Number : public Expression {

  // TODO: Use a variant here instead to store different number types
  std::string value;

  bool isNeg;
  bool isFloat;

  Number(reader::LocationRange &loc, const std::string &num, bool isNeg,
         bool isFloat)
      : Expression(loc), value(num), isNeg(isNeg), isFloat(isFloat){};

  ExprType getType() const override;
  std::string toString() const override;

  MaybeNode analyze(SereneContext &ctx) override;
  void generateIR(serene::Namespace &, mlir::ModuleOp &) override;

  // TODO: This is horrible, we need to fix it after the mvp
  int toI64() const;

  ~Number() = default;

  static bool classof(const Expression *e);
};

} // namespace exprs
} // namespace serene

#endif
