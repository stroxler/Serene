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

#ifndef SERENE_JIT_LAYERS_H
#define SERENE_JIT_LAYERS_H

#include "serene/reader/location.h"
#include "serene/utils.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/Layer.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/Support/Error.h>

#define LAYER_LOG(...) \
  DEBUG_WITH_TYPE("layer", llvm::dbgs() << "[Layer]: " << __VA_ARGS__ << "\n");

namespace orc = llvm::orc;

namespace serene {
class SereneContext;
class Namespace;

namespace exprs {
class Expression;
using Node = std::shared_ptr<Expression>;
using Ast  = std::vector<Node>;
} // namespace exprs

namespace jit {

class SereneAstLayer;

/// This will compile the ast to llvm ir.
llvm::orc::ThreadSafeModule compileAst(serene::SereneContext &ctx,
                                       exprs::Ast &ast);

class SerenAstMaterializationUnit : public orc::MaterializationUnit {
public:
  SerenAstMaterializationUnit(SereneContext &ctx, SereneAstLayer &l,
                              exprs::Ast &ast);

  llvm::StringRef getName() const override {
    return "SereneAstMaterializationUnit";
  }

  void
  materialize(std::unique_ptr<orc::MaterializationResponsibility> r) override;

private:
  void discard(const orc::JITDylib &jd,
               const orc::SymbolStringPtr &sym) override {
    UNUSED(jd);
    UNUSED(sym);
    llvm_unreachable("Serene functions are not overridable");
  }

  serene::SereneContext &ctx;
  SereneAstLayer &astLayer;
  exprs::Ast &ast;
};

// class SereneAstLayer {
//   serene::SereneContexe &ctx;
//   orc::IRLayer &baseLayer;
//   orc::MangleAndInterner &mangler;
//   const llvm::DataLayout &dl;

// public:
//   SereneAstLayer(serene::SereneContexe &ctx, orc::IRLayer &baseLayer,
//                  orc::MangleAndInterner &mangler, const llvm::DataLayout &dl)
//       : ctx(ctx), baseLayer(baseLayer), mangler(mangler), dl(dl){};

//   llvm::Error add(orc::ResourceTrackerSP &rt, exprs::Ast &ast) {
//     return rt->getJITDylib().define(
//         std::make_unique<SerenAstMaterializationUnit>(ctx, *this, ast), rt);
//   }

//   void emit(std::unique_ptr<orc::MaterializationResponsibility> mr,
//             exprs::Ast &e) {
//     baseLayer.emit(std::move(mr), compileAst(ctx, e));
//   }

//   orc::SymbolFlagsMap getInterface(exprs::Ast &e) {
//     orc::SymbolFlagsMap Symbols;
//     Symbols[mangler(e.getName())] = llvm::JITSymbolFlags(
//         llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable);
//     return Symbols;
//   }
// };

/// NS Layer ==================================================================
class NSLayer;

/// This will compile the NS to llvm ir.
llvm::orc::ThreadSafeModule compileNS(serene::SereneContext &ctx,
                                      serene::Namespace &ns);

class NSMaterializationUnit : public orc::MaterializationUnit {
public:
  NSMaterializationUnit(SereneContext &ctx, NSLayer &l, serene::Namespace &ns);

  llvm::StringRef getName() const override { return "NSMaterializationUnit"; }

  void
  materialize(std::unique_ptr<orc::MaterializationResponsibility> r) override;

private:
  void discard(const orc::JITDylib &jd,
               const orc::SymbolStringPtr &sym) override {
    UNUSED(jd);
    UNUSED(sym);
    UNUSED(ctx);
    // TODO: Check the ctx to see whether we need to remove the sym or not
  }

  serene::SereneContext &ctx;
  NSLayer &nsLayer;
  serene::Namespace &ns;
};

/// NS Layer is responsible for adding namespaces to the JIT
class NSLayer {
  serene::SereneContext &ctx;
  orc::IRLayer &baseLayer;
  orc::MangleAndInterner &mangler;
  const llvm::DataLayout &dl;

public:
  NSLayer(serene::SereneContext &ctx, orc::IRLayer &baseLayer,
          orc::MangleAndInterner &mangler, const llvm::DataLayout &dl)
      : ctx(ctx), baseLayer(baseLayer), mangler(mangler), dl(dl){};

  llvm::Error add(orc::ResourceTrackerSP &rt, llvm::StringRef nsname) {
    auto loc = serene::reader::LocationRange::UnknownLocation(nsname);

    return add(rt, nsname, loc);
  }

  llvm::Error add(orc::ResourceTrackerSP &rt, llvm::StringRef nsname,
                  serene::reader::LocationRange &loc);

  void emit(std::unique_ptr<orc::MaterializationResponsibility> mr,
            serene::Namespace &ns) {
    // TODO: We need to pass dl to the compilerNS later to aviod recreating
    // the data layout all the time
    UNUSED(dl);
    LAYER_LOG("Emit namespace");
    baseLayer.emit(std::move(mr), compileNS(ctx, ns));
  }

  orc::SymbolFlagsMap getInterface(serene::Namespace &ns);
};
} // namespace jit
} // namespace serene
#endif
