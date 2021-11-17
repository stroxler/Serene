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

#include "serene/exprs/call.h"

#include "serene/errors/error.h"
#include "serene/exprs/def.h"
#include "serene/exprs/expression.h"
#include "serene/exprs/list.h"
#include "serene/exprs/symbol.h"
#include "serene/namespace.h"
#include "serene/utils.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>

namespace serene {
namespace exprs {

ExprType Call::getType() const { return ExprType::Call; };

std::string Call::toString() const {
  return llvm::formatv("<Call {0} {1}>", this->target->toString(),
                       astToString(&this->params));
}

MaybeNode Call::analyze(semantics::AnalysisState &state) {
  UNUSED(state);

  return EmptyNode;
};

bool Call::classof(const Expression *e) {
  return e->getType() == ExprType::Call;
};

MaybeNode Call::make(semantics::AnalysisState &state, List *list) {

  // TODO: replace this with a runtime check
  assert((list->count() != 0) && "Empty call? Seriously ?");

  // Let's find out what is the first element of the list
  auto maybeFirst = list->elements[0]->analyze(state);

  if (!maybeFirst) {
    // There's something wrong with the first element. Return the error
    return maybeFirst;
  }

  Node first = maybeFirst.getValue();

  // No rewrite is needed for the first element
  if (!first) {
    first = list->elements[0];
  }

  Node targetNode;
  Ast rawParams;

  if (list->count() > 1) {
    rawParams = list->from(1);
  }

  // We need to create the Call node based on the type of the first
  // element after it being analyzed.
  switch (first->getType()) {

    // In case of a Symbol, We should look it up in the current scope and
    // if it resolves to a value. Then we have to make sure that the
    // return value is callable.
  case ExprType::Symbol: {

    auto *sym = llvm::dyn_cast<Symbol>(first.get());

    if (sym == nullptr) {
      llvm_unreachable("Couldn't case to Symbol while the type is symbol!");
    }

    auto maybeResult = state.env.lookup(sym->name);

    if (!maybeResult.hasValue()) {
      std::string msg =
          llvm::formatv("Can't resolve the symbol '{0}'", sym->name);
      return makeErrorful<Node>(sym->location, errors::CantResolveSymbol, msg);
    }

    targetNode = std::move(maybeResult.getValue());
    break;
  }

  case ExprType::Def:

    // If the first element was a Call itself we need to just chain it
    // with a new call. It would be something like `((blah 1) 4)`. `blah`
    // should return a callable expression itself, which we need to let
    // the typechecker to check
  case ExprType::Call:
    // If the first element was a function, then just use it as the target
    // of the call. It would be like `((fn (x) x) 4)`
  case ExprType::Fn: {
    targetNode = first;
    break;
  }

    // Otherwise we don't know how to call the first element.
  default: {
    std::string msg = llvm::formatv("Don't know how to call a '{0}'",
                                    stringifyExprType(first->getType()));
    return makeErrorful<Node>(first->location, errors::DontKnowHowToCallNode,
                              msg);
  }
  };

  auto analyzedParams = semantics::analyze(state, rawParams);

  if (!analyzedParams) {
    return MaybeNode::error(analyzedParams.getError());
  }

  return makeSuccessfulNode<Call>(list->location, targetNode,
                                  analyzedParams.getValue());
};
} // namespace exprs
} // namespace serene
