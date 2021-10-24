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

#ifndef SERENE_CONTEXT_H
#define SERENE_CONTEXT_H

#include "serene/diagnostics.h"
#include "serene/environment.h"
#include "serene/export.h"
#include "serene/namespace.h"
#include "serene/passes.h"
#include "serene/slir/dialect.h"
#include "serene/source_mgr.h"

#include <llvm/ADT/None.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Host.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>

#include <memory>

namespace serene {

namespace reader {
class LocationRange;
} // namespace reader

namespace exprs {
class Expression;
using Node = std::shared_ptr<Expression>;
} // namespace exprs

enum class CompilationPhase {
  Parse,
  Analysis,
  SLIR,
  MLIR, // Lowered slir to other dialects
  LIR,  // Lowered to the llvm ir dialect
  IR,   // Lowered to the LLVMIR itself
  NoOptimization,
  O1,
  O2,
  O3,
};

class SERENE_EXPORT SereneContext {
  struct Options {
    /// Whether to use colors for the output or not
    bool withColors = true;

    Options() = default;
  };

public:
  // --------------------------------------------------------------------------
  // IMPORTANT:
  // These two contextes have to be the very first members of the class in
  // order to destroy last. DO NOT change the order or add anything before
  // them
  // --------------------------------------------------------------------------
  llvm::LLVMContext llvmContext;
  mlir::MLIRContext mlirContext;

  mlir::PassManager pm;

  std::unique_ptr<DiagnosticEngine> diagEngine;

  /// The source manager is responsible for loading namespaces and practically
  /// managing the source code in form of memory buffers.
  SourceMgr sourceManager;

  /// The set of options to change the compilers behaivoirs
  Options opts;

  std::string targetTriple;

  /// Insert the given `ns` into the context. The Context object is
  /// the owner of all the namespaces. The `ns` will overwrite any
  /// namespace with the same name.
  void insertNS(NSPtr &ns);

  /// Sets the n ame of the current namespace in the context and return
  /// a boolean indicating the status of this operation. The operation
  /// will fail if the namespace does not exist in the namespace table.
  bool setCurrentNS(llvm::StringRef ns_name);

  /// Return the current namespace that is being processed at the moment
  Namespace &getCurrentNS();

  /// Lookup the namespace with the give name in the current context and
  /// return a pointer to it or a `nullptr` in it doesn't exist.
  Namespace *getNS(llvm::StringRef ns_name);

  SereneContext()
      : pm(&mlirContext), diagEngine(makeDiagnosticEngine(*this)),
        targetPhase(CompilationPhase::NoOptimization) {
    mlirContext.getOrLoadDialect<serene::slir::SereneDialect>();
    mlirContext.getOrLoadDialect<mlir::StandardOpsDialect>();

    // We need to create one empty namespace, so that the JIT can
    // start it's operation.
    auto ns = makeNamespace(*this, "serene.user", llvm::None);

    // TODO: Get the crash report path dynamically from the cli
    // pm.enableCrashReproducerGeneration("/home/lxsameer/mlir.mlir");

    // TODO: Set the target triple with respect to the CLI args
    targetTriple = llvm::sys::getDefaultTargetTriple();
  };

  /// Set the target compilation phase of the compiler. The compilation
  /// phase dictates the behavior and the output type of the compiler.
  void setOperationPhase(CompilationPhase phase);

  CompilationPhase getTargetPhase() { return targetPhase; };
  int getOptimizatioLevel();

  MaybeNS readNamespace(const std::string &name);
  MaybeNS readNamespace(const std::string &name, reader::LocationRange loc);

private:
  CompilationPhase targetPhase;

  // The namespace table. Every namespace that needs to be compiled has
  // to register itself with the context and appear on this table.
  // This table acts as a cache as well.
  std::map<std::string, NSPtr> namespaces;

  // Why string vs pointer? We might rewrite the namespace and
  // holding a pointer means that it might point to the old version
  std::string current_ns;
};

/// Creates a new context object. Contexts are used through out the compilation
/// process to store the state
SERENE_EXPORT std::unique_ptr<SereneContext> makeSereneContext();

/// Terminates the serene compiler process in a thread safe manner
SERENE_EXPORT void terminate(SereneContext &ctx, int exitCode);

} // namespace serene

#endif
