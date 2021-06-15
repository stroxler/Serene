
/*
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

#include "serene/exprs/number.h"

#include "serene/slir/dialect.h"
#include "serene/slir/utils.h"

namespace serene {
namespace exprs {

ExprType Number::getType() const { return ExprType::Number; };

std::string Number::toString() const {
  return llvm::formatv("<Number {0}>", value);
}

MaybeNode Number::analyze(SereneContext &ctx) {
  return MaybeNode::success(nullptr);
};

bool Number::classof(const Expression *e) {
  return e->getType() == ExprType::Number;
};

int Number::toI64() { return std::stoi(this->value); };

void Number::generateIR(serene::Namespace &ns) {
  mlir::OpBuilder builder(&ns.getContext().mlirContext);
  mlir::ModuleOp &module = ns.getModule();

  auto op = builder.create<serene::slir::ValueOp>(
      serene::slir::toMLIRLocation(ns, location.start), toI64());

  if (op) {
    module.push_back(op);
  }
  // TODO: in case of failure attach the error to the NS
};
} // namespace exprs
} // namespace serene
