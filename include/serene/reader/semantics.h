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

#ifndef READER_SEMANTICS_H
#define READER_SEMANTICS_H

#include "serene/context.h"
#include "serene/errors/error.h"
#include "serene/exprs/expression.h"

namespace serene::reader {
using AnalyzeResult = Result<exprs::Ast, std::vector<exprs::ErrorPtr>>;
/// This function is the entrypoint to the Semantic Analysis phase of **Serene**
/// It will call the `analyze` method on every node in the given AST and
/// returns a new AST as the result of the semantic analysis.
///
/// \param ctx The serene context
/// \prama tree The raw AST to analyze
AnalyzeResult analyze(serene::SereneContext &ctx, exprs::Ast &tree);
}; // namespace serene::reader

#endif
