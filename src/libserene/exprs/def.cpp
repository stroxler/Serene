/*
 * Serene Programming Language
 *
 * Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "serene/exprs/def.h"

#include "serene/errors/error.h"
#include "serene/exprs/expression.h"
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
  UNUSED(ctx);
  return EmptyNode;
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

  // TODO: Replace this one with a runtime check
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

  // TODO: To refactor this logic into a generic function
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

void Def::generateIR(serene::Namespace &ns, mlir::ModuleOp &m) {

  if (value->getType() == ExprType::Fn) {
    value->generateIR(ns, m);
    return;
  }
  m.emitError("Def: not implemented!");
};
} // namespace exprs
} // namespace serene
