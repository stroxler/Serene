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

#include "serene/slir/slir.h"

namespace serene {
namespace slir {

llvm::Optional<llvm::orc::ThreadSafeModule>
compileToLLVMIR(serene::SereneContext &ctx, mlir::ModuleOp &module) {

  auto llvmContext = serene::SereneContext::genLLVMContext();
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerLLVMDialectTranslation(*module.getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  auto llvmModule = mlir::translateModuleToLLVMIR(module, *llvmContext);

  if (!llvmModule) {
    // TODO: Return a Result type instead
    module.emitError("Failed to emit LLVM IR\n");
    return llvm::None;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // TODO: replace this call with our own version of setupTargetTriple
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/ctx.getOptimizatioLevel(), /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  if (auto err = optPipeline(llvmModule.get())) {
    // TODO: Return a proper error
    module.emitError("Failed to optimize LLVM IR\n");
    return llvm::None;
  }

  auto tsm = llvm::orc::ThreadSafeModule(std::move(llvmModule),
                                         std::move(llvmContext));
  return tsm;
};

} // namespace slir

} // namespace serene
