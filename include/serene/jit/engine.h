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

/**
 * Commentary:
 */

// TODO: Look at the LLJITBuilderState::prepareForConstruction to setup the
//       object linking layer

#ifndef SERENE_JIT_ENGINE_H
#define SERENE_JIT_ENGINE_H

#include "serene/jit/layers.h"
#include "serene/namespace.h"

#include <llvm/ExecutionEngine/Orc/CompileUtils.h> // for ConcurrentIRCompiler
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/EPCIndirectionUtils.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h> // for DynamicLibrarySearchGenerator
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/Mangling.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/Support/raw_ostream.h>

#include <memory>

#define JIT_LOG(...) \
  DEBUG_WITH_TYPE("jit", llvm::dbgs() << "[JIT]: " << __VA_ARGS__ << "\n");

namespace orc = llvm::orc;

namespace serene {

class SereneContext;

namespace jit {

class SereneJIT {
  /// An ExecutionSession represents a running JIT program.
  std::unique_ptr<orc::ExecutionSession> es;

  std::unique_ptr<orc::EPCIndirectionUtils> epciu;

  llvm::DataLayout dl;

  /// Mangler in responsible for changing the symbol names based on our
  /// naming convention.
  orc::MangleAndInterner mangler;

  // Layers -------------------------------------------------------------------
  // Serene's JIT follows the same design as the ORC and uses the framework that
  // it provides.
  //
  // Layers are the building blocks of the JIT and work on top of each other
  // to add different functionalities to the JIT. Order is important here and
  // layers call each other and pa

  /// The object linking layer allows object files to be added to the JIT
  orc::RTDyldObjectLinkingLayer objectLayer;

  /// The compiler layer is responsible to compile the LLVMIR to target code
  /// for us
  orc::IRCompileLayer compileLayer;

  // TODO: Enable these two layers when we add the `Platform support`
  // std::unique_ptr<orc::IRTransformLayer> transformLayer;
  // std::unique_ptr<orc::IRTransformLayer> initHelperTransformLayer;

  // orc::CompileOnDemandLayer compileOnDemandLayer;

  /// Transform layaer is responsible for running a pass pipeline on the AST
  /// and generate LLVM IR
  orc::IRTransformLayer transformLayer;

  /// The AST Layer reads and import the Serene Ast directly to the JIT
  // SereneAstLayer astLayer;

  /// NS layer is responsible for adding namespace to the JIT by name.
  /// It will import the entire namespace.
  NSLayer nsLayer;

  /// This a symbol table tha holds symbols from whatever code we execute
  orc::JITDylib &mainJD;

  serene::SereneContext &ctx;

  /// The name of the NS which we are in it at the moment.
  std::string inNS;

  static llvm::Expected<orc::ThreadSafeModule>
  optimizeModule(orc::ThreadSafeModule tsm,
                 const orc::MaterializationResponsibility &r) {
    // TSM.withModuleDo([](Module &M) {
    //   // Create a function pass manager.
    //   auto FPM = std::make_unique<legacy::FunctionPassManager>(&M);

    //   // Add some optimizations.
    //   FPM->add(createInstructionCombiningPass());
    //   FPM->add(createReassociatePass());
    //   FPM->add(createGVNPass());
    //   FPM->add(createCFGSimplificationPass());
    //   FPM->doInitialization();

    //   // Run the optimizations over all functions in the module being added
    //   to
    //   // the JIT.
    //   for (auto &F : M)
    //     FPM->run(F);
    // });
    UNUSED(r);
    return std::move(tsm);
  }

public:
  SereneJIT(serene::SereneContext &ctx,
            std::unique_ptr<orc::ExecutionSession> es,
            std::unique_ptr<orc::EPCIndirectionUtils> epciu,
            orc::JITTargetMachineBuilder jtmb, llvm::DataLayout &&dl);

  ~SereneJIT() {
    // if (compileThreads) {
    //   compileThreads->wait();
    // }

    if (auto err = es->endSession()) {
      es->reportError(std::move(err));
    }
  };

  const llvm::DataLayout &getDataLayout() const { return dl; }

  orc::JITDylib &getMainJITDylib() { return mainJD; }

  llvm::Error addModule(orc::ThreadSafeModule tsm,
                        orc::ResourceTrackerSP rt = nullptr) {
    if (!rt) {
      rt = mainJD.getDefaultResourceTracker();
    }

    return compileLayer.add(rt, std::move(tsm));
  }

  llvm::Error addNS(llvm::StringRef nsname,
                    orc::ResourceTrackerSP rt = nullptr);

  llvm::Expected<llvm::JITEvaluatedSymbol> lookup(llvm::StringRef name) {
    JIT_LOG("Looking up symbol: " + name);
    return es->lookup({&mainJD}, mangler(name.str()));
  }
};

llvm::Expected<std::unique_ptr<SereneJIT>>
makeSereneJIT(serene::SereneContext &ctx);

}; // namespace jit
}; // namespace serene

#endif
