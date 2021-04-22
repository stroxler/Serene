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

#include "serene/exprs/def.h"
#include "serene/errors/error.h"
#include "serene/exprs/list.h"
#include "serene/exprs/symbol.h"
#include "llvm/Support/FormatVariadic.h"

namespace serene {
namespace exprs {

ExprType Def::getType() const { return ExprType::Def; };

std::string Def::toString() const {
  return llvm::formatv("<Def {0} -> {1}>", this->binding,
                       this->value->toString());
}

maybe_node Def::analyze(reader::SemanticContext &ctx) {
  return Result<node>::success(nullptr);
};

bool Def::classof(const Expression *e) {
  return e->getType() == ExprType::Def;
};

std::shared_ptr<errors::Error> Def::isValid(List *list) {
  // TODO: Add support for docstring as the 3rd argument (4th element)
  if (list->count() != 3) {
    std::string msg = llvm::formatv("Expected 3 got {}", list->count());
    return makeAndCast<errors::Error>(&errors::DefWrongNumberOfArgs,
                                      list->elements[0], msg);
  }

  Symbol *binding = llvm::dyn_cast<Symbol>(list->elements[1].get());

  if (!binding) {
    return makeAndCast<errors::Error>(&errors::DefExpectSymbol,
                                      list->elements[1], "");
  }

  return nullptr;
};
} // namespace exprs
} // namespace serene
