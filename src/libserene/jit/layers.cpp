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

#include "serene/jit/layers.h"

#include "serene/context.h"
#include "serene/exprs/fn.h"
#include "serene/exprs/traits.h"

#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Support/Error.h> // for report_fatal_error

#include <algorithm>

namespace serene::jit {

llvm::orc::ThreadSafeModule compileAst(Namespace &ns, exprs::Ast &ast) {

  assert(ns.getTree().size() < ast.size() && "Did you add the ast to the NS?");

  LAYER_LOG("Compile in context of namespace: " + ns.name);
  unsigned offset = ns.getTree().size() - ast.size();

  auto maybeModule = ns.compileToLLVMFromOffset(offset);

  if (!maybeModule) {
    // TODO: Handle failure
    llvm::report_fatal_error("Couldn't compile lazily JIT'd function");
  }

  return std::move(maybeModule.getValue());
};

AstMaterializationUnit::AstMaterializationUnit(Namespace &ns, AstLayer &l,
                                               exprs::Ast &ast)
    : orc::MaterializationUnit(l.getInterface(ns, ast)), ns(ns), astLayer(l),
      ast(ast){};

void AstMaterializationUnit::materialize(
    std::unique_ptr<orc::MaterializationResponsibility> r) {
  astLayer.emit(std::move(r), ns, ast);
}

orc::MaterializationUnit::Interface AstLayer::getInterface(Namespace &ns,
                                                           exprs::Ast &e) {
  orc::SymbolFlagsMap Symbols;
  auto symList   = ns.getSymList();
  unsigned index = symList.size();

  // This probably will change symList
  auto err = ns.addTree(e);

  if (err) {
    // TODO: Fix this by a call to diag engine or return the err
    llvm::outs() << "Fixme: semantic err\n";
    return orc::MaterializationUnit::Interface(std::move(Symbols), nullptr);
  }

  auto &env            = ns.getRootEnv();
  auto populateTableFn = [&env, this, &Symbols](auto name) {
    auto flags     = llvm::JITSymbolFlags::Exported;
    auto maybeExpr = env.lookup(name.str());

    if (!maybeExpr) {
      LAYER_LOG("Skiping '" + name + "' symbol");
      return;
    }

    auto expr = maybeExpr.getValue();

    if (expr->getType() == exprs::ExprType::Fn) {
      flags = flags | llvm::JITSymbolFlags::Callable;
    }

    auto mangledSym = this->mangler(name);
    LAYER_LOG("Mangle symbol for: " + name + " = " << mangledSym);
    Symbols[mangledSym] = llvm::JITSymbolFlags(flags);
  };

  std::for_each(symList.begin() + index, symList.end(), populateTableFn);
  return orc::MaterializationUnit::Interface(std::move(Symbols), nullptr);
}

/// NS Layer ==================================================================

llvm::orc::ThreadSafeModule compileNS(Namespace &ns) {
  LAYER_LOG("Compile namespace: " + ns.name);

  auto maybeModule = ns.compileToLLVM();

  if (!maybeModule) {
    // TODO: Handle failure
    llvm::report_fatal_error("Couldn't compile lazily JIT'd function");
  }

  return std::move(maybeModule.getValue());
};

NSMaterializationUnit::NSMaterializationUnit(NSLayer &l, Namespace &ns)
    : MaterializationUnit(l.getInterface(ns)), nsLayer(l), ns(ns){};

void NSMaterializationUnit::materialize(
    std::unique_ptr<orc::MaterializationResponsibility> r) {
  nsLayer.emit(std::move(r), ns);
}

llvm::Error NSLayer::add(orc::ResourceTrackerSP &rt, llvm::StringRef nsname,
                         reader::LocationRange &loc) {

  LAYER_LOG("Add namespace: " + nsname);
  auto maybeNS = ctx.readNamespace(nsname.str(), loc);

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
      std::make_unique<NSMaterializationUnit>(*this, *ns), rt);
}

orc::MaterializationUnit::Interface
NSLayer::getInterface(serene::Namespace &ns) {
  orc::SymbolFlagsMap Symbols;

  for (auto &k : ns.getRootEnv()) {
    auto flags = llvm::JITSymbolFlags::Exported;
    auto name  = k.getFirst();
    auto expr  = k.getSecond();

    if (expr->getType() == exprs::ExprType::Fn) {
      flags = flags | llvm::JITSymbolFlags::Callable;
    }

    auto mangledSym = mangler(name);
    LAYER_LOG("Mangle symbol for: " + name + " = " << mangledSym);
    Symbols[mangledSym] = llvm::JITSymbolFlags(flags);
  }

  return orc::MaterializationUnit::Interface(std::move(Symbols), nullptr);
}

} // namespace serene::jit
