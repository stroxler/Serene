/* -*- C++ -*-
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

#include "serene/jit/layers.h"

#include "serene/context.h"
#include "serene/exprs/fn.h"
#include "serene/exprs/traits.h"

#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Support/Error.h> // for report_fatal_error

namespace serene::jit {

// llvm::orc::ThreadSafeModule compileAst(serene::SereneContext &ctx,
//                                        exprs::Ast &ast){

// };

// SerenAstMaterializationUnit::SerenAstMaterializationUnit(
//     serene::SereneContext &ctx, SereneAstLayer &l, exprs::Ast &ast)
//     : MaterializationUnit(l.getInterface(ast), nullptr), ctx(ctx),
//     astLayer(l),
//       ast(ast){};

// void SerenAstMaterializationUnit::materialize(
//     std::unique_ptr<orc::MaterializationResponsibility> r) {
//   astLayer.emit(std::move(r), ast);
// }

/// NS Layer ==================================================================

llvm::orc::ThreadSafeModule compileNS(serene::SereneContext &ctx,
                                      serene::Namespace &ns) {
  UNUSED(ctx);

  LAYER_LOG("Compile namespace: " + ns.name);

  auto maybeModule = ns.compileToLLVM();

  if (!maybeModule) {
    // TODO: Handle failure
    llvm::report_fatal_error("Couldn't compile lazily JIT'd function");
  }

  return std::move(maybeModule.getValue());
};

NSMaterializationUnit::NSMaterializationUnit(SereneContext &ctx, NSLayer &l,
                                             serene::Namespace &ns)
    : MaterializationUnit(l.getInterface(ns), nullptr), ctx(ctx), nsLayer(l),
      ns(ns){};

void NSMaterializationUnit::materialize(
    std::unique_ptr<orc::MaterializationResponsibility> r) {
  nsLayer.emit(std::move(r), ns);
}

llvm::Error NSLayer::add(orc::ResourceTrackerSP &rt, llvm::StringRef nsname,
                         reader::LocationRange &loc) {

  LAYER_LOG("Add namespace: " + nsname);
  auto maybeNS = ctx.sourceManager.readNamespace(ctx, nsname.str(), loc);

  if (!maybeNS) {
    // TODO: Fix this by making Serene errors compatible with llvm::Error
    auto err = maybeNS.getError();
    return llvm::make_error<llvm::StringError>(
        llvm::Twine(err.front()->getMessage()),
        std::make_error_code(std::errc::io_error));
  }

  auto ns = maybeNS.getValue();

  LAYER_LOG("Add the materialize unit for: " + nsname);
  return rt->getJITDylib().define(
      std::make_unique<NSMaterializationUnit>(ctx, *this, *ns), rt);
}

orc::SymbolFlagsMap NSLayer::getInterface(serene::Namespace &ns) {
  orc::SymbolFlagsMap Symbols;

  for (auto &k : ns.getRootEnv()) {
    auto flags = llvm::JITSymbolFlags::Exported;
    auto name  = k.getFirst();
    auto expr  = k.getSecond();

    if (expr->getType() == exprs::ExprType::Fn) {
      flags = flags | llvm::JITSymbolFlags::Callable;
    }

    auto mangledSym = mangler(k.getFirst());
    LAYER_LOG("Mangle symbol for: " + k.getFirst() + " = " << mangledSym);
    Symbols[mangledSym] = llvm::JITSymbolFlags(flags);
  }

  return Symbols;
}

} // namespace serene::jit
