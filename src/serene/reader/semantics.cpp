/* -*- C++ -*-
 * Serene programming language.
 *
 *  Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "serene/reader/semantics.h"

#include "serene/context.h"
#include "serene/exprs/expression.h"

namespace serene::reader {

AnalyzeResult analyze(serene::SereneContext &ctx, exprs::Ast &inputAst) {
  // TODO: Fetch the current namespace from the JIT engine later and if it is
  // `nil` then the given `ast` has to start with a namespace definition.

  std::vector<exprs::ErrorPtr> errors;
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
