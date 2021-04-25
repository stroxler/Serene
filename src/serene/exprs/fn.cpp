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

#include "serene/exprs/fn.h"
#include "serene/errors/error.h"
#include "serene/exprs/expression.h"
#include "serene/exprs/list.h"
#include "serene/exprs/symbol.h"
#include "serene/reader/semantics.h"
#include "llvm/Support/FormatVariadic.h"

namespace serene {
namespace exprs {

ExprType Fn::getType() const { return ExprType::Fn; };

std::string Fn::toString() const {
  return llvm::formatv("<Fn {0} {1} to {2}>",
                       this->name.empty() ? "Anonymous" : this->name,
                       this->args.toString(),
                       this->body.empty() ? "<>" : astToString(&this->body));
}

MaybeNode Fn::analyze(SereneContext &ctx) {
  return Result<Node>::success(nullptr);
};

bool Fn::classof(const Expression *e) { return e->getType() == ExprType::Fn; };

MaybeNode Fn::make(SereneContext &ctx, List *list) {
  // TODO: Add support for docstring as the 3rd argument (4th element)
  if (list->count() < 2) {
    std::string msg =
        llvm::formatv("The argument list is mandatory.", list->count());
    return Result<Node>::success(makeAndCast<errors::Error>(
        &errors::FnNoArgsList, list->elements[0], msg));
  }

  Symbol *fnSym = llvm::dyn_cast<Symbol>(list->elements[0].get());
  assert((fnSym && fnSym->name == "fn") &&
         "The first element of the list should be a 'fn'.");

  List *args = llvm::dyn_cast<List>(list->elements[1].get());

  if (!args) {
    std::string msg =
        llvm::formatv("Arguments of a function has to be a list, got '{0}'",
                      stringifyExprType(list->elements[1]->getType()));
    return Result<Node>::success(makeAndCast<errors::Error>(
        &errors::FnArgsMustBeList, list->elements[1], msg));
  }

  Ast body;

  if (list->count() > 2) {
    body = std::vector<Node>(list->begin() + 2, list->end());
    auto maybeAst = reader::analyze(ctx, body);

    if (!maybeAst) {
      return Result<Node>::error(std::move(maybeAst.getError()));
    }

    body = maybeAst.getValue();
  }

  Node fn = exprs::make<Fn>(list->location, *args, body);
  return Result<Node>::success(fn);
};
} // namespace exprs
} // namespace serene
