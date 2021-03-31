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

#include "serene/sir/generator.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "serene/expr.hpp"
#include "serene/sir/dialect.hpp"

namespace serene {
namespace sir {

mlir::ModuleOp Generator::generate() {
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

  for (auto x : ns.Tree()) {
    module.push_back(generateExpression(x.get()));
  }

  return module;
};

mlir::Operation *Generator::generateExpression(AExpr *x) {
  switch (x->getType()) {
  case SereneType::Number: {
    return generateNumber(llvm::cast<Number>(x));
  }

  case SereneType::List: {
    return generateList(llvm::cast<List>(x));
  }

  default: {
    return builder.create<ValueOp>(builder.getUnknownLoc(), (uint64_t)3);
  }
  }
};

mlir::Operation *Generator::generateList(List *l) {
  auto first = l->at(0);

  if (!first) {
    // Empty list.
    // TODO: Return Nil or empty list.

    // Just for now.
    return builder.create<ValueOp>(builder.getUnknownLoc(), (uint64_t)0);
  }

  // for (auto x : l->from(1)) {
  //   generateExpression(x);
  // }
  return builder.create<ValueOp>(builder.getUnknownLoc(), (uint64_t)0);
};

mlir::Operation *Generator::generateNumber(Number *x) {
  return builder.create<ValueOp>(builder.getUnknownLoc(), x->toI64());
};

Generator::~Generator(){};
} // namespace sir

} // namespace serene
