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
#include "serene/exprs/expression.h"

namespace serene::reader {

SereneContext makeSereneContext() { return SereneContext(); }
/// The entry point to the Semantic analysis phase. It calls the `analyze`
/// method of each node in the given AST and creates a new AST that contains a
/// more comprehensive set of nodes in a semantically correct AST. If the
/// `analyze` method of a node return a `nullptr` value as the `success` result
/// (Checkout the `Result` type in `utils.h`) then the original node will be
/// used instead. Also please note that in **Serene** Semantic errors
/// represented as AST nodes as well. So you should expect an `analyze` method
/// of a node to return a `Result<node>::Success(Error...)` in case of a
/// semantic error.
/// \param ctx The semantic analysis context
/// \param inputAst The raw AST to analyze and possibly rewrite.
exprs::maybe_ast analyze(SereneContext &ctx, exprs::ast &inputAst) {
  // TODO: Fetch the current namespace from the JIT engine later and if it is
  // `nil` then the given `ast` has to start with a namespace definition.

  exprs::ast ast;

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
      Result<exprs::ast>::error(std::move(maybeNode.getError()));
    }
  }

  return Result<exprs::ast>::success(std::move(ast));
};
}; // namespace serene::reader
