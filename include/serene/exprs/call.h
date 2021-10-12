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

#ifndef EXPRS_CALL_H
#define EXPRS_CALL_H

#include "serene/context.h"
#include "serene/errors/error.h"
#include "serene/exprs/expression.h"
#include "serene/exprs/list.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <memory>
#include <string>

namespace serene {

namespace exprs {
class List;

/// This data structure represents a function. with a collection of
/// arguments and the ast of a body
class Call : public Expression {

public:
  Node target;
  Ast params;

  Call(reader::LocationRange &loc, Node &target, Ast &params)
      : Expression(loc), target(target), params(params){};

  Call(Call &) = delete;

  ExprType getType() const override;
  std::string toString() const override;
  MaybeNode analyze(SereneContext &) override;
  void generateIR(serene::Namespace &, mlir::ModuleOp &) override{};

  static bool classof(const Expression *e);

  /// Creates a call node out of a list.
  /// For exmaple: `(somefn (param1 param2) param3)`. This function
  /// is supposed to be used in the semantic analysis phase.
  ///
  /// \param ctx The semantic analysis context object.
  /// \param list the list in question.

  static MaybeNode make(SereneContext &ctx, List *list);

  ~Call() = default;
};

} // namespace exprs
} // namespace serene

#endif
