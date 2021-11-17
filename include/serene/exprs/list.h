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

#ifndef SERENE_EXPRS_LIST_H
#define SERENE_EXPRS_LIST_H

#include "serene/context.h"
#include "serene/exprs/expression.h"

#include <llvm/ADT/Optional.h>

#include <string>

namespace serene {

namespace exprs {

/// This class represents a List in the AST level and not the List as the data
/// type.
class List : public Expression {
public:
  // Internal elements of the lest (small vector of shared pointers to
  // expressions)
  Ast elements;

  List(const List &l);               // Copy ctor
  List(List &&e) noexcept = default; // Move ctor

  List(const reader::LocationRange &loc) : Expression(loc){};
  List(const reader::LocationRange &loc, Node &e);
  List(const reader::LocationRange &loc, Ast elems);

  ExprType getType() const override;
  std::string toString() const override;

  void append(Node);

  size_t count() const;

  Ast from(uint index);

  llvm::Optional<Expression *> at(uint index);

  /// Return an iterator to be used with the `for` loop. It's implicitly called
  /// by the for loop.
  std::vector<Node>::const_iterator cbegin();

  /// Return an iterator to be used with the `for` loop. It's implicitly called
  /// by the for loop.
  std::vector<Node>::const_iterator cend();

  /// Return an iterator to be used with the `for` loop. It's implicitly called
  /// by the for loop.
  std::vector<Node>::iterator begin();

  /// Return an iterator to be used with the `for` loop. It's implicitly called
  /// by the for loop.
  std::vector<Node>::iterator end();

  MaybeNode analyze(semantics::AnalysisState &state) override;
  // NOLINTNEXTLINE(readability-named-parameter)
  void generateIR(serene::Namespace &, mlir::ModuleOp &) override{};

  ~List() = default;

  static bool classof(const Expression *e);
};

} // namespace exprs
} // namespace serene

#endif
