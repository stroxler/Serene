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

#include "serene/exprs/expression.h"

#include <llvm/Support/FormatVariadic.h>

namespace serene {
namespace exprs {

std::string astToString(const Ast *tree) {
  if (tree->size() == 0) {
    return "";
  }

  std::string result = tree->at(0)->toString();

  for (unsigned int i = 1; i < tree->size(); i++) {
    result = llvm::formatv("{0} {1}", result, tree->at(i)->toString());
  }

  return result;
}

std::string stringifyExprType(ExprType t) { return exprTypes[(int)t]; };

/// Dump the given AST tree to the standard out
void dump(Ast &tree) { llvm::outs() << astToString(&tree) << "\n"; };

} // namespace exprs
} // namespace serene
