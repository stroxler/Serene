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

#include "serene/exprs/fn.h"

#include "serene/errors.h"
#include "serene/exprs/expression.h"
#include "serene/exprs/list.h"
#include "serene/exprs/symbol.h"
#include "serene/slir/dialect.h"
#include "serene/slir/utils.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <utility>

namespace serene {
namespace exprs {

Fn::Fn(SereneContext &ctx, reader::LocationRange &loc, List &args, Ast body)
    : Expression(loc), args(args), body(std::move(body)) {
  this->setName(
      llvm::formatv("___fn___{0}", ctx.getCurrentNS().nextFnCounter()));
};

ExprType Fn::getType() const { return ExprType::Fn; };

std::string Fn::toString() const {
  return llvm::formatv("<Fn {0} {1} to {2}>",
                       this->name.empty() ? "Anonymous" : this->name,
                       this->args.toString(),
                       this->body.empty() ? "<>" : astToString(&this->body));
}

MaybeNode Fn::analyze(semantics::AnalysisState &state) {
  UNUSED(state);

  return EmptyNode;
};

bool Fn::classof(const Expression *e) { return e->getType() == ExprType::Fn; };

void Fn::setName(std::string n) { this->name = std::move(n); };

MaybeNode Fn::make(semantics::AnalysisState &state, List *list) {

  auto &ctx = state.ns.getContext();

  // TODO: Add support for docstring as the 3rd argument (4th element)
  if (list->count() < 2) {
    return errors::makeError(ctx, errors::FnNoArgsList,
                             list->elements[0]->location,
                             "The argument list is mandatory.");
  }

  Symbol *fnSym = llvm::dyn_cast<Symbol>(list->elements[0].get());

  // TODO: Replace this assert with a runtime check
  assert((fnSym && fnSym->name == "fn") &&
         "The first element of the list should be a 'fn'.");

  List *args = llvm::dyn_cast<List>(list->elements[1].get());

  if (args == nullptr) {
    std::string msg =
        llvm::formatv("Arguments of a function has to be a list, got '{0}'",
                      stringifyExprType(list->elements[1]->getType()));

    return errors::makeError(ctx, errors::FnArgsMustBeList,
                             list->elements[1]->location, msg);
  }

  Ast body;

  // If there is a body for this function analyze the body and set
  // the retuned ast as the final body
  if (list->count() > 2) {
    body = std::vector<Node>(list->begin() + 2, list->end());

    // TODO: call state.moveToNewEnv to create a new env and set the
    //       arguments into the new env.

    auto maybeAst = semantics::analyze(state, body);

    if (!maybeAst) {
      return maybeAst.takeError();
    }

    body = *maybeAst;
  }

  return makeSuccessfulNode<Fn>(ctx, list->location, *args, body);
};

void Fn::generateIR(serene::Namespace &ns, mlir::ModuleOp &m) {
  auto loc  = slir::toMLIRLocation(ns, location.start);
  auto &ctx = ns.getContext();

  mlir::OpBuilder builder(&ctx.mlirContext);

  // llvm::SmallVector<mlir::Type, 4> arg_types;
  llvm::SmallVector<mlir::NamedAttribute, 4> arguments;
  // at the moment we only support integers
  for (auto &arg : args) {
    auto *argSym = llvm::dyn_cast<Symbol>(arg.get());

    if (argSym == nullptr) {
      m->emitError(llvm::formatv(
          "Arguments of a function have to be symbols. Fn: '{0}'", name));
      return;
    }

    arguments.push_back(builder.getNamedAttr(
        argSym->name, mlir::TypeAttr::get(builder.getI64Type())));
  }

  auto fn = builder.create<slir::Fn1Op>(
      loc, builder.getI64Type(), name,
      mlir::DictionaryAttr::get(builder.getContext(), arguments),
      builder.getStringAttr("public"));

  if (!fn) {
    m.emitError(llvm::formatv("Can't create the function '{0}'", name));
    return;
  }

  auto &body       = fn.body();
  auto *entryBlock = new mlir::Block();

  body.push_back(entryBlock);

  builder.setInsertionPointToStart(entryBlock);
  auto retVal = builder.create<slir::Value1Op>(loc, 0).getResult();

  slir::Return1Op returnOp = builder.create<slir::Return1Op>(loc, retVal);

  if (!returnOp) {
    m.emitError(
        llvm::formatv("Can't create the return value of function '{0}'", name));
    fn.erase();
    return;
  }

  m.push_back(fn);
};
} // namespace exprs
} // namespace serene
