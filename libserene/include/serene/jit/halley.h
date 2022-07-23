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

/**
 * Commentary:
  This is the first working attempt on building a JIT engine for Serene
  and named after Edmond Halley.

  - It operates in lazy (for REPL) and non-lazy mode and wraps LLJIT
    and LLLazyJIT
  - It uses an object cache layer to cache module (not NSs) objects.
 */

// TODO: [jit] When we want to load any static or dynamic lib for
// namespace as a dependency first look up the `ExecutionSession`
// to make sure that we did not load it already. If we did just
// use the existing `JITDylib` for it.

//
// TODO: [jit] Use Bare JITDylibs for the static and dynamic libs.
// Hint: Look at `createBareJITDylib` on the `ExecutionSession`

#ifndef SERENE_JIT_HALLEY_H
#define SERENE_JIT_HALLEY_H

#include "serene/context.h" // for Serene...
#include "serene/export.h"  // for SERENE...
#include "serene/fs.h"
#include "serene/types/types.h" // for Intern...

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/None.h>
#include <llvm/ADT/SmallVector.h>                             // for SmallV...
#include <llvm/ADT/StringMap.h>                               // for StringMap
#include <llvm/ADT/StringRef.h>                               // for StringRef
#include <llvm/ExecutionEngine/ObjectCache.h>                 // for Object...
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h> // for JITTar...
#include <llvm/ExecutionEngine/Orc/LLJIT.h>                   // for LLJIT
#include <llvm/Support/Debug.h>                               // for dbgs
#include <llvm/Support/Error.h>                               // for Expected
#include <llvm/Support/MemoryBuffer.h>                        // for Memory...
#include <llvm/Support/MemoryBufferRef.h>                     // for Memory...
#include <llvm/Support/raw_ostream.h>                         // for raw_os...

#include <memory>   // for unique...
#include <stddef.h> // for size_t
#include <vector>   // for vector

#define HALLEY_LOG(...)                  \
  DEBUG_WITH_TYPE("halley", llvm::dbgs() \
                                << "[HALLEY]: " << __VA_ARGS__ << "\n");

#define MAIN_PROCESS_JD_NAME "<process>"

namespace llvm {
class DataLayout;
class JITEventListener;
class Module;
namespace orc {
class JITDylib;
} // namespace orc

} // namespace llvm

namespace serene {
namespace jit {
class Halley;

// Why? This is the lazy man's way to make it easier to replace
// the class under the hood later on to test different implementaion
// with the same interface
using Engine             = Halley;
using EnginePtr          = std::unique_ptr<Engine>;
using MaybeEngine        = llvm::Expected<EnginePtr>;
using MaybeJitAddress    = llvm::Expected<void *(*)()>;
using Dylib              = llvm::orc::JITDylib;
using DylibPtr           = Dylib *;
using MaybeDylibPtr      = llvm::Expected<DylibPtr>;
using MaybeNSFileTypeArr = llvm::Optional<llvm::ArrayRef<fs::NSFileType>>;

/// A simple object cache following Lang's LLJITWithObjectCache example and
/// MLIR's SimpelObjectCache.
class ObjectCache : public llvm::ObjectCache {
public:
  /// Cache the given `objBuffer` for the given module `m`. The buffer contains
  /// the combiled objects of the module
  void notifyObjectCompiled(const llvm::Module *m,
                            llvm::MemoryBufferRef objBuffer) override;

  // Lookup the cache for the given module `m` or returen a nullptr.
  std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module *m) override;

  /// Dump cached object to output file `filename`.
  void dumpToObjectFile(llvm::StringRef filename);

private:
  llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> cachedObjects;
};

class SERENE_EXPORT Halley {
  // TODO: Replace this with a variant of LLJIT and LLLazyJIT
  std::unique_ptr<llvm::orc::LLJIT> engine;
  std::unique_ptr<ObjectCache> cache;
  /// GDB notification listener.
  llvm::JITEventListener *gdbListener;
  /// Perf notification listener.
  llvm::JITEventListener *perfListener;
  llvm::orc::JITTargetMachineBuilder jtmb;
  // TODO: [cleanup][jit] Since we can access to the data layout via
  // `engine.getDataLayout`, remove this attribute and it's usecases
  llvm::DataLayout &dl;
  // /TODO

  std::unique_ptr<SereneContext> ctx;
  bool isLazy = false;

  // TODO: [jit] Replace this vector with a thread safe time-optimized
  // datastructure that is capable of indexing strings and own all
  // the strings. A lockless algorithm would be even better

  /// Owns all the internal strings used in the compilation process
  std::vector<types::InternalString *> stringStorage;

  std::vector<types::Namespace *> nsStorage;
  // /TODO

  // JIT JITDylib related functions ---
  llvm::StringMap<llvm::SmallVector<llvm::orc::JITDylib *, 1>> jitDylibs;

  /// Register the given pointer to a `JITDylib` \p l, with the give \p ns.
  void pushJITDylib(types::Namespace &ns, llvm::orc::JITDylib *l);

  // /// Returns the number of registered `JITDylib` for the given \p ns.
  size_t getNumberOfJITDylibs(types::Namespace &ns);

  types::Namespace &makeNamespace(const char *name);

  // ==========================================================================
  // Loading namespaces from different sources like source files, objectfiles
  // etc
  // ==========================================================================
  struct NSLoadRequest {
    llvm::StringRef nsName;
    llvm::StringRef path;
    std::string &nsToFileName;
  };

  /// This function loads the namespace by the given `nsName` from the file
  /// in the given `path`. It assumes that the `path` exists.
  MaybeDylibPtr loadNamespaceFrom(fs::NSFileType type_, NSLoadRequest &req);

  template <fs::NSFileType fileType>
  MaybeDylibPtr loadNamespaceFrom(NSLoadRequest &req);
  // ==========================================================================

  std::vector<const char *> getContainedNamespaces(llvm::StringRef name,
                                                   DylibPtr jd);

  llvm::Error createCurrentProcessJD();

public:
  Halley(std::unique_ptr<SereneContext> ctx,
         llvm::orc::JITTargetMachineBuilder &&jtmb, llvm::DataLayout &&dl);

  // TODO: [jit] Create a function to "require" a namespace as a dependency.
  // If the namespace already exists return it otherwise call `loadNamespace`.

  /// Load a namespace by exploring the load paths and different file
  /// formats to find the namespace. We assume that we want to load
  /// the namespace from file even if it exists already.
  MaybeDylibPtr loadNamespace(std::string &nsName);

  // TODO: Move all the loader related functions to a Loader class
  /// Load the shared library in the given `path` to the given JITDylib
  /// `jd` via the give ExecutionSession `es`.
  /// This function assumes that the shared lib exists.
  MaybeDylibPtr loadSharedLibFile(llvm::StringRef name, llvm::StringRef path);
  MaybeDylibPtr loadStaticLibrary(const std::string &name);
  MaybeDylibPtr loadSharedLibrary(const std::string &name);
  // /TODO

  static MaybeEngine make(std::unique_ptr<SereneContext> sereneCtxPtr,
                          llvm::orc::JITTargetMachineBuilder &&jtmb);

  SereneContext &getContext() { return *ctx; };

  llvm::Error createEmptyNS(const char *name);
  const types::InternalString &getInternalString(const char *s);

  /// Return a pointer to the most registered JITDylib of the given \p ns
  ////name
  llvm::orc::JITDylib *getLatestJITDylib(const types::Namespace &ns);
  llvm::orc::JITDylib *getLatestJITDylib(const char *nsName);

  void setEngine(std::unique_ptr<llvm::orc::LLJIT> e, bool isLazy);
  /// Looks up a packed-argument function with the given sym name and returns a
  /// pointer to it. Propagates errors in case of failure.
  MaybeJitAddress lookup(const char *nsName, const char *sym);
  MaybeJitAddress lookup(const types::Symbol &sym) const;

  /// Invokes the function with the given name passing it the list of opaque
  /// pointers to the actual arguments.
  // llvm::Error
  // invokePacked(llvm::StringRef name,
  //              llvm::MutableArrayRef<void *> args = llvm::None) const;

  /// Trait that defines how a given type is passed to the JIT code. This
  /// defaults to passing the address but can be specialized.
  template <typename T>
  struct Argument {
    static void pack(llvm::SmallVectorImpl<void *> &args, T &val) {
      args.push_back(&val);
    }
  };

  /// Tag to wrap an output parameter when invoking a jitted function.
  template <typename T>
  struct FnResult {
    explicit FnResult(T &result) : value(result) {}
    T &value;
  };

  /// Helper function to wrap an output operand when using
  /// ExecutionEngine::invoke.
  template <typename T>
  static FnResult<T> result(T &t) {
    return FnResult<T>(t);
  }

  // Specialization for output parameter: their address is forwarded directly to
  // the native code.
  template <typename T>
  struct Argument<FnResult<T>> {
    static void pack(llvm::SmallVectorImpl<void *> &args, FnResult<T> &result) {
      args.push_back(&result.value);
    }
  };

  llvm::Error loadModule(const char *nsName, const char *file);
  void dumpToObjectFile(llvm::StringRef filename);
};

MaybeEngine makeHalleyJIT(std::unique_ptr<SereneContext> ctx);

} // namespace jit
} // namespace serene

#endif
