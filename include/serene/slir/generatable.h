/* -*- C++ -*-
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

#ifndef SERENE_SLIR_GENERATABLE_H
#define SERENE_SLIR_GENERATABLE_H

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

  void generate(serene::Namespace &ns) {
    // TODO: should we return any status or possible error here or
    //       should we just populate them in a ns wide state?
    this->Object().generateIR(ns);
  };
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
  llvm::InitializeNativeTargetAsmPrinter();
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
