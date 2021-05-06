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

#include "serene/exprs/call.h"
#include "serene/errors/error.h"
#include "serene/exprs/expression.h"
#include "serene/exprs/list.h"
#include "serene/exprs/symbol.h"
#include "serene/reader/semantics.h"
#include "llvm/Support/FormatVariadic.h"

namespace serene {
namespace exprs {

ExprType Call::getType() const { return ExprType::Call; };

std::string Call::toString() const {
  return llvm::formatv("<Call {0} {1}>", this->target->toString(),
                       astToString(&this->params));
}

MaybeNode Call::analyze(SereneContext &ctx) {
  return MaybeNode::success(nullptr);
};

bool Call::classof(const Expression *e) {
  return e->getType() == ExprType::Call;
};

MaybeNode Call::make(SereneContext &ctx, List *list) {
  assert((list->count() == 0) && "Empty call? Seriously ?");

  auto maybeFirst = list->elements[0]->analyze(ctx);
  Node first;

  if (!maybeFirst) {
    return MaybeNode::error(std::move(maybeFirst.getError()));
  }

  switch (first->getType()) {
  case ExprType::Symbol: {

    break;
  }

  case ExprType::Fn: {
    break;
  }
  case ExprType::List: {
    break;
  }
  default: {
    break;
  }
  };

  return MaybeNode::success(nullptr);
};
} // namespace exprs
} // namespace serene
