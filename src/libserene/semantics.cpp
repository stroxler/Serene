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

#include "serene/semantics.h"

#include "serene/context.h"
#include "serene/exprs/expression.h"
#include "serene/namespace.h"

#include <llvm/Support/Error.h>

namespace serene::semantics {

std::unique_ptr<AnalysisState> AnalysisState::moveToNewEnv() {
  auto &newEnv = ns.createEnv(&env);
  return makeAnalysisState(ns, newEnv);
};

AnalyzeResult analyze(AnalysisState &state, exprs::Ast &forms) {
  llvm::Error errors = llvm::Error::success();
  exprs::Ast ast;

  for (auto &element : forms) {
    auto maybeNode = element->analyze(state);

    // Is it a `success` result
    if (maybeNode) {
      auto &node = *maybeNode;

      if (node) {
        // is there a new node to replace the current node ?
        ast.push_back(node);
      } else {
        // Analyze returned a `nullptr`. No rewrite is needed.
        // Use the current element instead.
        ast.push_back(element);
      }
    } else {

      // `analyze` returned an errorful result. This type of error
      // is llvm related and has to be raised later
      // (std::move());
      auto err = maybeNode.takeError();
      errors   = llvm::joinErrors(std::move(errors), std::move(err));
    }
  }

  // If the errors (which is an ErrorList) contains error and is
  // not succssful
  if (!errors) {
    return std::move(ast);
  }

  return std::move(errors);
};
}; // namespace serene::semantics
