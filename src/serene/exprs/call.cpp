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
#include "serene/exprs/def.h"
#include "serene/exprs/expression.h"
#include "serene/exprs/list.h"
#include "serene/exprs/symbol.h"
#include "serene/reader/semantics.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

namespace serene {
namespace exprs {

ExprType Call::getType() const { return ExprType::Call; };

std::string Call::toString() const {
  return llvm::formatv("<Call {0} {1}>", this->target->toString(),
                       astToString(&this->params));
}

MaybeNode Call::analyze(SereneContext &ctx) { return EmptyNode; };

bool Call::classof(const Expression *e) {
  return e->getType() == ExprType::Call;
};

MaybeNode Call::make(SereneContext &ctx, List *list) {
  assert((list->count() != 0) && "Empty call? Seriously ?");

  // Let's find out what is the first element of the list
  auto maybeFirst = list->elements[0]->analyze(ctx);

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

    if (!sym) {
      llvm_unreachable("Couldn't case to Symbol while the type is symbol!");
    }
    // TODO: Lookup the symbol in the namespace via a method that looks
    //       into the current environment.
    auto maybeResult = ctx.getCurrentNS()->semanticEnv.lookup(sym->name);

    if (!maybeResult.hasValue()) {
      std::string msg =
          llvm::formatv("Can't resolve the symbol '{0}'", sym->name);
      return makeErrorful<Node>(sym->location, &errors::CantResolveSymbol, msg);
    }

    targetNode = maybeResult.getValue();
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
    return makeErrorful<Node>(first->location, &errors::DontKnowHowToCallNode,
                              msg);
  }
  };

  auto analyzedParams = reader::analyze(ctx, rawParams);

  if (!analyzedParams) {
    return MaybeNode::error(analyzedParams.getError());
  }

  return makeSuccessfulNode<Call>(list->location, targetNode,
                                  analyzedParams.getValue());
};
} // namespace exprs
} // namespace serene
