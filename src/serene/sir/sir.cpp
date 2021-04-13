/**
 * Serene programming language.
 *
 *  Copyright (c) 2020 Sameer Rahmani <lxsameer@gnu.org>
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

#include "serene/sir/sir.hpp"
#include "mlir/IR/MLIRContext.h"
#include "serene/exprs/expression.h"
#include "serene/sir/dialect.hpp"
#include "serene/sir/generator.hpp"
#include <memory>

namespace serene {
namespace sir {
SIR::SIR() { context.getOrLoadDialect<::serene::sir::SereneDialect>(); }

mlir::OwningModuleRef SIR::generate(::serene::Namespace *ns) {
  auto g = std::make_unique<Generator>(context, ns);

  return g->generate();
};

SIR::~SIR() {}

void dumpSIR(exprs::ast &t) {
  auto ns = new ::serene::Namespace("user", llvm::None);

  SIR s{};

  if (failed(ns->setTree(t))) {
    llvm::errs() << "Can't set the body of the namespace";
    return;
  }

  auto module = s.generate(ns);
  module->dump();
};

} // namespace sir
} // namespace serene
