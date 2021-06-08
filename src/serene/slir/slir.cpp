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

#include "serene/slir/slir.h"
#include "mlir/IR/MLIRContext.h"
#include "serene/exprs/expression.h"
#include "serene/namespace.h"
#include "serene/slir/dialect.h"
#include "serene/slir/generator.h"
#include <memory>

namespace serene {
namespace slir {

SLIR::SLIR(serene::SereneContext &ctx) : context(ctx) {
  context.mlirContext.getOrLoadDialect<serene::slir::SereneDialect>();
}

mlir::OwningModuleRef SLIR::generate(llvm::StringRef ns_name) {
  auto g = std::make_unique<Generator>(context, ns_name);

  return g->generate();
};

SLIR::~SLIR() {}

void dumpSLIR(serene::SereneContext &ctx, llvm::StringRef ns_name) {
  SLIR s(ctx);
  auto ns = ctx.getNS(ns_name);

  assert(!ns && "No such a namespace to dump!");

  auto module = s.generate(ns_name);
  module->dump();
};

} // namespace slir
} // namespace serene
