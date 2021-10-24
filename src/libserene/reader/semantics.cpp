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

#include "serene/reader/semantics.h"

#include "serene/context.h"
#include "serene/exprs/expression.h"

namespace serene::reader {

AnalyzeResult analyze(serene::SereneContext &ctx, exprs::Ast &inputAst) {
  // TODO: Fetch the current namespace from the JIT engine later and if it is
  // `nil` then the given `ast` has to start with a namespace definition.

  errors::ErrorTree errors;
  exprs::Ast ast;

  for (auto &element : inputAst) {
    auto maybeNode = element->analyze(ctx);

    // Is it a `success` result
    if (maybeNode) {
      auto &node = maybeNode.getValue();

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
      auto errVector = maybeNode.getError();
      errors.insert(errors.end(), errVector.begin(), errVector.end());
    }
  }

  if (errors.empty()) {
    return AnalyzeResult::success(std::move(ast));
  }

  return AnalyzeResult::error(std::move(errors));
};
}; // namespace serene::reader
