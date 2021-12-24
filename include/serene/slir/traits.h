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

#ifndef SERENE_SLIR_TRAITS_H
#define SERENE_SLIR_TRAITS_H

#include "serene/slir/dialect.h"
#include "serene/traits.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

#include <stdexcept>
#include <utility>

namespace serene {
class Namespace;
class SereneContext;
} // namespace serene

namespace serene::slir {

template <typename T>
class GeneratableUnit : public TraitBase<T, GeneratableUnit> {
public:
  GeneratableUnit(){};
  GeneratableUnit(const GeneratableUnit &) = delete;

  void generate(serene::Namespace &ns) { this->Object().generateIR(ns); };
};

template <typename T>
class Generatable : public TraitBase<T, Generatable> {
public:
  Generatable(){};
  Generatable(const Generatable &) = delete;

  mlir::LogicalResult generate() { return this->Object().generate(); };
  mlir::LogicalResult runPasses() { return this->Object().runPasses(); };

  mlir::ModuleOp &getModule() { return this->Object().getModule(); };
  serene::SereneContext &getContext() { return this->Object().getContext(); };

  void dump() { this->Object().dump(); };
};

template <typename T>
mlir::LogicalResult generate(Generatable<T> &t) {
  return t.generate();
};

template <typename T>
std::unique_ptr<llvm::Module> toLLVMIR(Generatable<T> &t) {
  auto &module = t.getModule();
  auto &ctx    = t.getContext();
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerLLVMDialectTranslation(ctx.mlirContext);

  // Convert the module to LLVM IR in a new LLVM IR context.
  auto llvmModule = mlir::translateModuleToLLVMIR(module, ctx.llvmContext);
  if (!llvmModule) {
    // TODO: Return a Result type instead
    llvm::errs() << "Failed to emit LLVM IR\n";
    throw std::runtime_error("Failed to emit LLVM IR\n");
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  // llvm::InitializeNativeTargetAsmPrinter();

  // TODO: replace this call with our own version of setupTargetTriple
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/ctx.getOptimizatioLevel(), /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    throw std::runtime_error("Failed to optimize LLVM IR");
  }

  return std::move(llvmModule);
};

template <typename T>
void dump(Generatable<T> &t) {
  t.dump();
};

} // namespace serene::slir

#endif
