/*
 * Serene Programming Language
 *
 * Copyright (c) 2019-2022 Sameer Rahmani <lxsameer@gnu.org>
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

#include "serene/errors.h"
#include "serene/exprs/expression.h"
#include "serene/exprs/fn.h"
#include "serene/exprs/list.h"
#include "serene/exprs/symbol.h"
#include "serene/exprs/traits.h"
#include "serene/slir/dialect.h"
#include "serene/slir/utils.h"

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

MaybeNode Def::analyze(semantics::AnalysisState &state) {
  UNUSED(state);

  return EmptyNode;
};

bool Def::classof(const Expression *e) {
  return e->getType() == ExprType::Def;
};

MaybeNode Def::make(semantics::AnalysisState &state, List *list) {
  auto &ctx = state.ns.getContext();

  // TODO: Add support for docstring as the 3rd argument (4th element)
  if (list->count() != 3) {
    std::string msg = llvm::formatv("Expected 3 got {0}", list->count());
    return errors::makeError(ctx, errors::DefWrongNumberOfArgs,
                             list->elements[0]->location, msg);
  }

  // Make sure that the list starts with a `def`
  Symbol *defSym = llvm::dyn_cast<Symbol>(list->elements[0].get());

  assert((defSym && defSym->name == "def") &&
         "The first element of the list should be a 'def'.");

  // Make sure that the first argument is a Symbol
  Symbol *binding = llvm::dyn_cast<Symbol>(list->elements[1].get());
  if (binding == nullptr) {
    return errors::makeError(ctx, errors::DefExpectSymbol,
                             list->elements[1]->location);
  }

  // Analyze the value
  MaybeNode value = list->elements[2]->analyze(state);
  Node analyzedValue;

  // TODO: To refactor this logic into a generic function
  if (value) {
    // Success value
    auto &valueNode = *value;

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
    if (tmp == nullptr) {
      llvm_unreachable("inconsistent getType for function");
    }

    tmp->setName(binding->name);
  }

  auto result = state.ns.define(binding->name, analyzedValue);

  if (result.succeeded()) {
    return makeSuccessfulNode<Def>(list->location, binding->name,
                                   analyzedValue);
  }
  llvm_unreachable("Inserting a value in the semantic env failed!");
};

void Def::generateIR(serene::Namespace &ns, mlir::ModuleOp &m) {

  if (value->getType() == ExprType::Fn) {
    value->generateIR(ns, m);
    return;
  }

  // auto loc   = slir::toMLIRLocation(ns, location.start);
  // auto &mctx = ns.getContext().mlirContext;

  // mlir::OpBuilder builder(&mctx);

  // auto sym = slir::SymbolType::get(&mctx, ns.name, binding);

  // TODO: we need to change the generate method of expressions
  //       to return mlir::Value or any wrapper of that which would
  //       be the ssa form of the result of the expression.
  //       and then use it to define the def op here.
  // auto def = builder.create<slir::DefOp>(sym, binding, value);
  m.emitError("Def: not implemented!");
};
} // namespace exprs
} // namespace serene
