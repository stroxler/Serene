/*
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

#include "serene/exprs/symbol.h"

#include "serene/exprs/expression.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/FormatVariadic.h>

namespace serene {
namespace exprs {

ExprType Symbol::getType() const { return ExprType::Symbol; };

std::string Symbol::toString() const {
  return llvm::formatv("<Symbol {0}/{1}>", nsName, name);
}

MaybeNode Symbol::analyze(semantics::AnalysisState &state) {
  UNUSED(state);

  return EmptyNode;
};

bool Symbol::classof(const Expression *e) {
  return e->getType() == ExprType::Symbol;
};

} // namespace exprs
} // namespace serene
