/* -*- C++ -*-
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

#include "serene/slir/generatable.h"

#include "serene/slir/dialect.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

#include <stdexcept>
#include <utility>

namespace serene {
namespace slir {

// mlir::Operation *Generatable::generate(exprs::Expression *x) {
//  switch (x->getType()) {
//  case SereneType::Number: {
//    return generate(llvm::cast<Number>(x));
//  }

// case SereneType::List: {
//   generate(llvm::cast<List>(x));
//   return nullptr;
// }

// default: {
// return builder.create<ValueOp>(builder.getUnknownLoc(), (uint64_t)3);
// }
// }
//};

// mlir::Value Generator::generate(exprs::List *l) {
//  auto first = l->at(0);

// if (!first) {
//   // Empty list.
//   // TODO: Return Nil or empty list.

//   // Just for now.
//   return builder.create<ValueOp>(builder.getUnknownLoc(), (uint64_t)0);
// }

// if (first->get()->getType() == SereneType::Symbol) {
//   auto fnNameSymbol = llvm::dyn_cast<Symbol>(first->get());

//   if (fnNameSymbol->getName() == "fn") {
//     if (l->count() <= 3) {
//       module.emitError("'fn' form needs exactly 2 arguments.");
//     }

//     auto args = llvm::dyn_cast<List>(l->at(1).getValue().get());
//     auto body = llvm::dyn_cast<List>(l->from(2).get());

//     if (!args) {
//       module.emitError("The first element of 'def' has to be a symbol.");
//     }

//     // Create a new anonymous function and push it to the anonymous
//     functions
//     // map, later on we
//     auto loc(fnNameSymbol->location->start);
//     auto anonymousName = fmt::format("__fn_{}__", anonymousFnCounter);
//     anonymousFnCounter++;

//     auto fn = generateFn(loc, anonymousName, args, body);
//     mlir::Identifier fnid = builder.getIdentifier(anonymousName);
//     anonymousFunctions.insert({fnid, fn});
//     return builder.create<FnIdOp>(builder.getUnknownLoc(), fnid.str());
//   }
// }
// // auto rest = l->from(1);
// // auto loc = toMLIRLocation(&first->get()->location->start);
// // for (auto x : *rest) {
// //   generate(x.get());
// // }
//  return builder.create<ValueOp>(builder.getUnknownLoc(), (uint64_t)100);
//};

// mlir::FuncOp Generator::generateFn(serene::reader::Location loc,
//                                    std::string name, List *args, List *body)
//                                    {

//   auto location = toMLIRLocation(&loc);
//   llvm::SmallVector<mlir::Type, 4> arg_types(args->count(),
//                                              builder.getI64Type());
//   auto func_type = builder.getFunctionType(arg_types, builder.getI64Type());
//   auto proto = mlir::FuncOp::create(location, name, func_type);
//   mlir::FuncOp fn(proto);

//   if (!fn) {
//     module.emitError("Can not create the function.");
//   }

//   auto &entryBlock = *fn.addEntryBlock();
//   llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>
//   scope(symbolTable);

//   // Declare all the function arguments in the symbol table.
//   for (const auto arg :
//        llvm::zip(args->asArrayRef(), entryBlock.getArguments())) {

//     auto argSymbol = llvm::dyn_cast<Symbol>(std::get<0>(arg).get());
//     if (!argSymbol) {
//       module.emitError("Function parameters must be symbols");
//     }
//     if (symbolTable.count(argSymbol->getName())) {
//       return nullptr;
//     }
//     symbolTable.insert(argSymbol->getName(), std::get<1>(arg));
//   }

//   // Set the insertion point in the builder to the beginning of the function
//   // body, it will be used throughout the codegen to create operations in
//   this
//   // function.
//   builder.setInsertionPointToStart(&entryBlock);

//   // Emit the body of the function.
//   if (!generate(body)) {
//     fn.erase();
//     return nullptr;
//   }

//   // // Implicitly return void if no return statement was emitted.
//   // // FIXME: we may fix the parser instead to always return the last
//   // expression
//   // // (this would possibly help the REPL case later)
//   // func::ReturnOp returnOp;

//   // if (!entryBlock.empty())
//   //   returnOp = dyn_cast<func::ReturnOp>(entryBlock.back());
//   // if (!returnOp) {
//   //   builder.create<func::ReturnOp>(loc(funcAST.getProto()->loc()));
//   // } else if (returnOp.hasOperand()) {
//   //   // Otherwise, if this return operation has an operand then add a
//   result
//   //   to
//   //   // the function.
//   // function.setType(builder.getFunctionType(function.getType().getInputs(),
//   //                                            getType(VarType{})));
//   // }

//   return fn;
// }
} // namespace slir

} // namespace serene
