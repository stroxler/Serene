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

#include "mlir/IR/BuiltinAttributes.h"
#include "serene/errors/error.h"
#include "serene/exprs/expression.h"
#include "serene/exprs/list.h"
#include "serene/exprs/symbol.h"
#include "serene/reader/semantics.h"
#include "serene/slir/dialect.h"
#include "serene/slir/utils.h"

#include <cstdint>
#include <llvm/Support/Casting.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/Block.h>

namespace serene {
namespace exprs {

Fn::Fn(SereneContext &ctx, reader::LocationRange &loc, List &args, Ast body)
    : Expression(loc), args(args), body(body) {
  this->setName(
      llvm::formatv("___fn___{0}", ctx.getCurrentNS()->nextFnCounter()));
};

ExprType Fn::getType() const { return ExprType::Fn; };

std::string Fn::toString() const {
  return llvm::formatv("<Fn {0} {1} to {2}>",
                       this->name.empty() ? "Anonymous" : this->name,
                       this->args.toString(),
                       this->body.empty() ? "<>" : astToString(&this->body));
}

MaybeNode Fn::analyze(SereneContext &ctx) { return EmptyNode; };

bool Fn::classof(const Expression *e) { return e->getType() == ExprType::Fn; };

void Fn::setName(std::string n) { this->name = n; };

MaybeNode Fn::make(SereneContext &ctx, List *list) {
  // TODO: Add support for docstring as the 3rd argument (4th element)
  if (list->count() < 2) {
    return makeErrorful<Node>(list->elements[0]->location,
                              &errors::FnNoArgsList,
                              "The argument list is mandatory.");
  }

  Symbol *fnSym = llvm::dyn_cast<Symbol>(list->elements[0].get());
  assert((fnSym && fnSym->name == "fn") &&
         "The first element of the list should be a 'fn'.");

  List *args = llvm::dyn_cast<List>(list->elements[1].get());

  if (!args) {
    std::string msg =
        llvm::formatv("Arguments of a function has to be a list, got '{0}'",
                      stringifyExprType(list->elements[1]->getType()));
    return makeErrorful<Node>(list->elements[1]->location,
                              &errors::FnArgsMustBeList, msg);
  }

  Ast body;

  if (list->count() > 2) {
    body          = std::vector<Node>(list->begin() + 2, list->end());
    auto maybeAst = reader::analyze(ctx, body);

    if (!maybeAst) {
      return MaybeNode::error(std::move(maybeAst.getError()));
    }

    body = maybeAst.getValue();
  }

  return makeSuccessfulNode<Fn>(ctx, list->location, *args, body);
};

void Fn::generateIR(serene::Namespace &ns) {
  auto loc     = slir::toMLIRLocation(ns, location.start);
  auto &ctx    = ns.getContext();
  auto &module = ns.getModule();
  mlir::OpBuilder builder(&ctx.mlirContext);

  // llvm::SmallVector<mlir::Type, 4> arg_types;
  llvm::SmallVector<mlir::NamedAttribute, 4> arguments;
  // at the moment we only support integers
  for (auto &arg : args) {
    auto *argSym = llvm::dyn_cast<Symbol>(arg.get());

    if (!argSym) {
      module->emitError(llvm::formatv(
          "Arguments of a function have to be symbols. Fn: '{0}'", name));
      return;
    }

    arguments.push_back(builder.getNamedAttr(
        argSym->name, mlir::TypeAttr::get(builder.getI64Type())));
  }

  // auto funcType = builder.getFunctionType(arg_types, builder.getI64Type());
  auto fn = builder.create<slir::FnOp>(
      loc, builder.getI64Type(), name,
      mlir::DictionaryAttr::get(builder.getContext(), arguments),
      builder.getStringAttr("public"));

  if (!fn) {
    module.emitError(llvm::formatv("Can't create the function '{0}'", name));
  }

  auto *entryBlock = new mlir::Block();
  auto &body       = fn.body();
  body.push_back(entryBlock);
  builder.setInsertionPointToStart(entryBlock);
  auto retVal = builder.create<slir::ValueOp>(loc, 0).getResult();

  mlir::ReturnOp returnOp = builder.create<mlir::ReturnOp>(loc, retVal);

  if (!returnOp) {
    module.emitError(
        llvm::formatv("Can't create the return value of function '{0}'", name));
    return;
  }
  module.push_back(fn);
};
} // namespace exprs
} // namespace serene
