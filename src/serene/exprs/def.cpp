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

maybe_node Def::make(reader::SemanticContext &ctx, List *list) {
  // TODO: Add support for docstring as the 3rd argument (4th element)

  if (list->count() != 3) {
    std::string msg = llvm::formatv("Expected 3 got {0}", list->count());
    return Result<node>::success(makeAndCast<errors::Error>(
        &errors::DefWrongNumberOfArgs, list->elements[0], msg));
  }

  // Make sure that the list starts with a `def`
  Symbol *defSym = llvm::dyn_cast<Symbol>(list->elements[0].get());
  assert((defSym && defSym->name == "def") &&
         "The first element of the list should be a 'def'.");

  // Make sure that the first argument is a Symbol
  Symbol *binding = llvm::dyn_cast<Symbol>(list->elements[1].get());
  if (!binding) {
    return Result<node>::success(makeAndCast<errors::Error>(
        &errors::DefExpectSymbol, list->elements[1], ""));
  }

  // Analyze the value
  maybe_node value = list->elements[2]->analyze(ctx);
  node analyzedValue;

  if (value) {
    // Success value
    auto &valueNode = value.getValue();

    if (valueNode) {
      // A rewrite is necessary
      analyzedValue = valueNode;
    } else {
      // no rewrite
      analyzedValue = list->elements[2];
    }
  } else {
    // Error value
    return value;
  }

  node def = exprs::make<Def>(list->location, binding->name, analyzedValue);
  return Result<node>::success(def);
};
} // namespace exprs
} // namespace serene
