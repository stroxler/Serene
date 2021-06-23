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
#include "serene/exprs/fn.h"
#include "serene/exprs/list.h"
#include "serene/exprs/symbol.h"
#include "serene/exprs/traits.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>

namespace serene {
namespace exprs {

ExprType Def::getType() const { return ExprType::Def; };

std::string Def::toString() const {
  return llvm::formatv("<Def {0} -> {1}>", this->binding,
                       this->value->toString());
}

MaybeNode Def::analyze(SereneContext &ctx) {
  return MaybeNode::success(nullptr);
};

bool Def::classof(const Expression *e) {
  return e->getType() == ExprType::Def;
};

MaybeNode Def::make(SereneContext &ctx, List *list) {
  // TODO: Add support for docstring as the 3rd argument (4th element)

  if (list->count() != 3) {
    std::string msg = llvm::formatv("Expected 3 got {0}", list->count());
    return makeErrorful<Node>(list->elements[0]->location,
                              &errors::DefWrongNumberOfArgs, msg);
  }

  // Make sure that the list starts with a `def`
  Symbol *defSym = llvm::dyn_cast<Symbol>(list->elements[0].get());
  assert((defSym && defSym->name == "def") &&
         "The first element of the list should be a 'def'.");

  // Make sure that the first argument is a Symbol
  Symbol *binding = llvm::dyn_cast<Symbol>(list->elements[1].get());
  if (!binding) {
    return makeErrorful<Node>(list->elements[1]->location,
                              &errors::DefExpectSymbol, "");
  }

  // Analyze the value
  MaybeNode value = list->elements[2]->analyze(ctx);
  Node analyzedValue;

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

  if (analyzedValue->getType() == ExprType::Fn) {
    Fn *tmp = llvm::dyn_cast<Fn>(analyzedValue.get());
    if (!tmp) {
      llvm_unreachable("inconsistent getType for function");
    }
    tmp->setName(binding->name);
  }
  // auto analayzedValuePtr = analyzedValue;
  auto result = ctx.getCurrentNS()->semanticEnv.insert_symbol(binding->name,
                                                              analyzedValue);

  if (result.succeeded()) {
    return makeSuccessfulNode<Def>(list->location, binding->name,
                                   analyzedValue);
  } else {
    llvm_unreachable("Inserting a value in the semantic env failed!");
  }
};
void Def::generateIR(serene::Namespace &ns) {
  auto &module = ns.getModule();

  if (value->getType() == ExprType::Fn) {
    value->generateIR(ns);
    return;
  }
  module.emitError("Def: not implemented!");
};
} // namespace exprs
} // namespace serene
