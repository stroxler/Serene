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

#include "serene/exprs/number.h"

#include "serene/exprs/expression.h"
#include "serene/slir/dialect.h"
#include "serene/slir/ops.h"
#include "serene/slir/utils.h"

namespace serene {
namespace exprs {

ExprType Number::getType() const { return ExprType::Number; };

std::string Number::toString() const {
  return llvm::formatv("<Number {0}>", value);
}

MaybeNode Number::analyze(semantics::AnalysisState &state) {
  UNUSED(state);

  return EmptyNode;
};

bool Number::classof(const Expression *e) {
  return e->getType() == ExprType::Number;
};

int Number::toI64() const { return std::stoi(this->value); };

void Number::generateIR(serene::Namespace &ns, mlir::ModuleOp &m) {
  mlir::OpBuilder builder(&ns.getContext().mlirContext);

  auto op = builder.create<serene::slir::Value1Op>(
      serene::slir::toMLIRLocation(ns, location.start), toI64());

  if (op) {
    m.push_back(op);
  }
  // TODO: in case of failure attach the error to the NS
};
} // namespace exprs
} // namespace serene
