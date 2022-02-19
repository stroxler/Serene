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

#ifndef SERENE_CONTEXT_H
#define SERENE_CONTEXT_H

#include "serene/diagnostics.h"
#include "serene/environment.h"
#include "serene/export.h"
#include "serene/jit/halley.h"
#include "serene/namespace.h"
#include "serene/passes.h"
#include "serene/slir/dialect.h"
#include "serene/source_mgr.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/None.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Triple.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Host.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>

#include <memory>

#define DEFAULT_NS_NAME "serene.user"
#define INTERNAL_NS     "serene.internal"

namespace serene {
class SereneContext;

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

/// Terminates the serene compiler process in a thread safe manner
SERENE_EXPORT void terminate(SereneContext &ctx, int exitCode);

struct SERENE_EXPORT Options {

  /// Whether to use colors for the output or not
  bool withColors = true;

  // JIT related flags
  bool JITenableObjectCache              = true;
  bool JITenableGDBNotificationListener  = true;
  bool JITenablePerfNotificationListener = true;
  bool JITLazy                           = false;

  Options() = default;
};

class SERENE_EXPORT SereneContext {

public:
  template <typename T>
  using CurrentNSFn = std::function<T()>;

  // --------------------------------------------------------------------------
  // IMPORTANT:
  // These two contextes have to be the very first members of the class in
  // order to destroy last. DO NOT change the order or add anything before
  // them
  // --------------------------------------------------------------------------

  // TODO: Remove the llvmContext
  llvm::LLVMContext llvmContext;
  mlir::MLIRContext mlirContext;

  mlir::PassManager pm;

  std::unique_ptr<DiagnosticEngine> diagEngine;

  std::unique_ptr<serene::jit::Halley> jit;

  /// The source manager is responsible for loading namespaces and practically
  /// managing the source code in form of memory buffers.
  SourceMgr sourceManager;

  /// The set of options to change the compilers behaivoirs
  Options opts;

  std::string targetTriple;

  // TODO: Replace target Triple with this one
  llvm::Triple triple;

  /// Insert the given `ns` into the context. The Context object is
  /// the owner of all the namespaces. The `ns` will overwrite any
  /// namespace with the same name.
  void insertNS(NSPtr &ns);

  /// Execute the given function \p f by setting the `currentNS`
  /// to the given \p nsName. It will restore the value of `currentNS`
  /// after \p f returned.
  template <typename T>
  T withCurrentNS(llvm::StringRef nsName, CurrentNSFn<T> f) {
    assert(!currentNS.empty() && "The currentNS is not initialized!");
    auto tmp        = this->currentNS;
    this->currentNS = nsName.str();

    T res           = f();
    this->currentNS = tmp;
    return res;
  };

  // void specialization
  template <>
  void withCurrentNS(llvm::StringRef nsName, CurrentNSFn<void> f) {
    assert(!currentNS.empty() && "The currentNS is not initialized!");
    auto tmp        = this->currentNS;
    this->currentNS = nsName.str();

    f();
    this->currentNS = tmp;
  }

  /// Return the current namespace that is being processed at the moment
  Namespace &getCurrentNS();

  /// Lookup the namespace with the give name in the current context and
  /// return a pointer to it or a `nullptr` in it doesn't exist.
  Namespace *getNS(llvm::StringRef nsName);

  /// Lookup and return a shared pointer to the given \p ns_name. This
  /// method should be used only if you need to own the namespace as well
  /// and want to keep it long term (like the JIT).
  NSPtr getSharedPtrToNS(llvm::StringRef nsName);

  SereneContext(Options &options)
      : pm(&mlirContext), diagEngine(makeDiagnosticEngine(*this)),
        opts(options), targetPhase(CompilationPhase::NoOptimization) {
    mlirContext.getOrLoadDialect<serene::slir::SereneDialect>();
    mlirContext.getOrLoadDialect<mlir::StandardOpsDialect>();

    // We need to create one empty namespace, so that the JIT can
    // start it's operation.
    auto ns = Namespace::make(*this, DEFAULT_NS_NAME, llvm::None);

    insertNS(ns);
    currentNS = ns->name;

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

  // Namespace stuff ---

  /// Create an empty namespace with the given \p name and optional \p filename
  /// and then insert it into the context
  NSPtr makeNamespace(llvm::StringRef name,
                      llvm::Optional<llvm::StringRef> filename);

  /// Read a namespace with the given \p name and returne a share pointer
  /// to the name or an Error tree.
  ///
  /// It just `read` the namespace by parsing it and running the semantic
  /// analyzer on it.
  MaybeNS readNamespace(const std::string &name);
  MaybeNS readNamespace(const std::string &name, reader::LocationRange loc);

  /// Reads and add the namespace with the given \p name to the context. The
  /// namespace will be added to the context and the JIT engine as well.
  ///
  /// It will \r a shared pointer to the namespace or an error tree.
  MaybeNS importNamespace(const std::string &name);
  MaybeNS importNamespace(const std::string &name, reader::LocationRange loc);
  // ---

  static std::unique_ptr<llvm::LLVMContext> genLLVMContext() {
    return std::make_unique<llvm::LLVMContext>();
  };

  static std::unique_ptr<SereneContext> make(Options &options) {
    auto ctx = std::make_unique<SereneContext>(options);
    auto *ns = ctx->getNS(DEFAULT_NS_NAME);

    assert(ns != nullptr && "Default ns doesn't exit!");

    auto maybeJIT = serene::jit::makeHalleyJIT(*ctx);

    if (!maybeJIT) {
      auto err = maybeJIT.takeError();
      panic(*ctx, err);
    }

    ctx->jit.swap(*maybeJIT);

    // Make serene.user which is the defult NS available on the
    // JIT
    auto loc = reader::LocationRange::UnknownLocation(INTERNAL_NS);
    auto err = ctx->jit->addNS(*ns, loc);

    // TODO: Fix this by calling to the diag engine
    if (err) {
      llvm::errs() << err << "\n";
      serene::terminate(*ctx, 1);
      return nullptr;
    }
    return ctx;
  };

  llvm::Triple getTargetTriple() const { return llvm::Triple(targetTriple); };

  // JIT JITDylib related functions ---

  // TODO: For Dylib related functions, make sure that the namespace in questoin
  //       is aleady registered in the context

  /// Return a pointer to the most registered JITDylib of the given \p ns
  ////name
  llvm::orc::JITDylib *getLatestJITDylib(Namespace &ns);

  /// Register the given pointer to a `JITDylib` \p l, with the give \p ns.
  void pushJITDylib(Namespace &ns, llvm::orc::JITDylib *l);

  /// Returns the number of registered `JITDylib` for the given \p ns.
  size_t getNumberOfJITDylibs(Namespace &ns);

private:
  CompilationPhase targetPhase;

  // TODO: Change it to a LLVM::StringMap
  // TODO: We need to keep different instances of the namespace
  //       because if any one of them gets cleaned up via reference
  //       count (if we are still using shared ptr for namespaces if not
  //       remove this todo) then we will end up with dangling references
  //       it the JIT

  // The namespace table. Every namespace that needs to be compiled has
  // to register itself with the context and appear on this table.
  // This table acts as a cache as well.
  std::map<std::string, NSPtr> namespaces;

  // Why string vs pointer? We might rewrite the namespace and
  // holding a pointer means that it might point to the old version
  std::string currentNS;

  /// A vector of pointers to all the jitDylibs for namespaces. Usually
  /// There will be only one pre NS but in case of forceful reloads of a
  /// namespace there will be more.
  llvm::StringMap<llvm::SmallVector<llvm::orc::JITDylib *, 1>> jitDylibs;
};

/// Creates a new context object. Contexts are used through out the compilation
/// process to store the state.
///
/// \p opts is an instance of \c Options that can be used to set options of
///         of the compiler.
SERENE_EXPORT std::unique_ptr<SereneContext>
makeSereneContext(Options opts = Options());

} // namespace serene

#endif
