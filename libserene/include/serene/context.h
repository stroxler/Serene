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

#include "serene/export.h" // for SERENE_EXPORT
#include "serene/options.h"

#include <llvm/ADT/Triple.h>     // for Triple
#include <llvm/ADT/Twine.h>      // for Twine
#include <llvm/IR/LLVMContext.h> // for LLVMContext
#include <llvm/Support/Host.h>   // for getDefaultTargetTriple

#include <functional> // for function
#include <memory>     // for make_unique, unique_ptr
#include <string>     // for string, basic_string
#include <vector>     // for vector

#define DEFAULT_NS_NAME "serene.user"
#define INTERNAL_NS     "serene.internal"

namespace serene {
class SereneContext;

/// This enum describes the different operational phases for the compiler
/// in order. Anything below `NoOptimization` is considered only for debugging
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
/// This function is only meant to be used in the compiler context
/// if you want to terminate the process in context of a serene program
/// via the JIT use an appropriate function in the `serene.core` ns.
SERENE_EXPORT void terminate(SereneContext &ctx, int exitCode);

class SERENE_EXPORT SereneContext {

public:
  template <typename T>
  using CurrentNSFn = std::function<T()>;

  /// The set of options to change the compilers behaivoirs
  Options opts;

  const llvm::Triple triple;

  explicit SereneContext(Options &options)
      : opts(options), triple(llvm::sys::getDefaultTargetTriple()),
        targetPhase(CompilationPhase::NoOptimization){};

  /// Set the target compilation phase of the compiler. The compilation
  /// phase dictates the behavior and the output type of the compiler.
  void setOperationPhase(CompilationPhase phase);

  CompilationPhase getTargetPhase() { return targetPhase; };
  int getOptimizatioLevel();

  static std::unique_ptr<llvm::LLVMContext> genLLVMContext() {
    return std::make_unique<llvm::LLVMContext>();
  };

  /// Setup the load path for namespace lookups
  void setLoadPaths(std::vector<std::string> &dirs) { loadPaths.swap(dirs); };

  // JIT JITDylib related functions ---

  // TODO: For Dylib related functions, make sure that the namespace in questoin
  //       is aleady registered in the context

  /// Return a pointer to the most registered JITDylib of the given \p ns
  ////name
  // llvm::orc::JITDylib *getLatestJITDylib(Namespace &ns);

  // /// Register the given pointer to a `JITDylib` \p l, with the give \p ns.
  // void pushJITDylib(Namespace &ns, llvm::orc::JITDylib *l);

  // /// Returns the number of registered `JITDylib` for the given \p ns.
  // size_t getNumberOfJITDylibs(Namespace &ns);

private:
  CompilationPhase targetPhase;
  std::vector<std::string> loadPaths;
  /// A vector of pointers to all the jitDylibs for namespaces. Usually
  /// There will be only one pre NS but in case of forceful reloads of a
  /// namespace there will be more.
  // llvm::StringMap<llvm::SmallVector<llvm::orc::JITDylib *, 1>> jitDylibs;
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
