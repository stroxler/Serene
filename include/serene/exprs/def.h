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

#ifndef SERENE_EXPRS_DEF_H
#define SERENE_EXPRS_DEF_H

#include "serene/context.h"
#include "serene/errors/error.h"
#include "serene/exprs/expression.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

#include <memory>
#include <string>

namespace serene {

namespace exprs {
class List;

/// This data structure represents the operation to define a new binding via
/// the `def` special form.
class Def : public Expression {

public:
  std::string binding;
  Node value;

  Def(reader::LocationRange &loc, llvm::StringRef binding, Node &v)
      : Expression(loc), binding(binding), value(v){};

  Def(Def &d) = delete;

  ExprType getType() const override;
  std::string toString() const override;

  MaybeNode analyze(semantics::AnalysisState &state) override;
  void generateIR(serene::Namespace &ns, mlir::ModuleOp &m) override;

  static bool classof(const Expression *e);

  /// Create a Def node out a list. The list should contain the
  /// correct `def` form like `(def blah value)`. This function
  /// is supposed to be used in the semantic analysis phase.
  ///
  /// \param state is the semantic analysis state to use in creation time.
  /// \param list the list containing the `def` form
  static MaybeNode make(semantics::AnalysisState &state, List *list);

  ~Def() = default;
};

} // namespace exprs
} // namespace serene

#endif
