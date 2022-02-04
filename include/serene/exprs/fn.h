/* -*- C++ -*-
 * Serene Programming Language
 *
 * Copyright (c) 2019-2022 Sameer Rahmani <lxsameer@gnu.org>
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

#ifndef SERENE_EXPRS_FN_H
#define SERENE_EXPRS_FN_H

#include "serene/context.h"
#include "serene/errors.h"
#include "serene/exprs/expression.h"
#include "serene/exprs/list.h"
#include "serene/namespace.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

#include <memory>
#include <string>

namespace serene {

namespace exprs {
class List;

/// This data structure represents a function. with a collection of
/// arguments and the ast of a body
class Fn : public Expression {

public:
  std::string name;

  // TODO: Use a coll type instead of a list here
  List args;
  Ast body;

  Fn(SereneContext &ctx, reader::LocationRange &loc, List &args, Ast body);

  Fn(Fn &f) = delete;

  ExprType getType() const override;
  std::string toString() const override;
  MaybeNode analyze(semantics::AnalysisState &state) override;
  void generateIR(serene::Namespace &ns, mlir::ModuleOp &m) override;

  static bool classof(const Expression *e);

  /// Creates a function node out of a function definition
  /// in a list. the list has to contain the correct definition
  /// of a function, for exmaple: `(fn (args1 arg2) body)`.This function
  /// is supposed to be used in the semantic analysis phase.
  ///
  /// \param state is the semantic analysis state to use.
  /// \param list the list containing the `fn` form
  static MaybeNode make(semantics::AnalysisState &state, List *list);

  void setName(std::string);
  ~Fn() = default;
};

} // namespace exprs
} // namespace serene

#endif
