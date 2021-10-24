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

#ifndef SERENE_JIT_H
#define SERENE_JIT_H

#include "serene/errors.h"
#include "serene/export.h"
#include "serene/exprs/expression.h"
#include "serene/namespace.h"
#include "serene/slir/generatable.h"
#include "serene/utils.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/Debug.h>
#include <mlir/Support/LLVM.h>

#include <memory>

#define JIT_LOG(...) \
  DEBUG_WITH_TYPE("JIT", llvm::dbgs() << "[JIT]: " << __VA_ARGS__ << "\n");

namespace serene {
class SERENE_EXPORT JIT;

using MaybeJIT = llvm::Optional<std::unique_ptr<JIT>>;

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

class JIT {
  // TODO: Should the JIT own the context ???
  Namespace &ns;

  std::unique_ptr<llvm::orc::LLJIT> engine;

  std::unique_ptr<ObjectCache> cache;

  /// GDB notification listener.
  llvm::JITEventListener *gdbListener;

  /// Perf notification listener.
  llvm::JITEventListener *perfListener;

public:
  JIT(Namespace &ns, bool enableObjectCache = true,
      bool enableGDBNotificationListener  = true,
      bool enablePerfNotificationListener = true);

  static MaybeJIT
  make(Namespace &ns, mlir::ArrayRef<llvm::StringRef> sharedLibPaths = {},
       mlir::Optional<llvm::CodeGenOpt::Level> jitCodeGenOptLevel = llvm::None,
       bool enableObjectCache = true, bool enableGDBNotificationListener = true,
       bool enablePerfNotificationListener = true);

  /// Looks up a packed-argument function with the given name and returns a
  /// pointer to it.  Propagates errors in case of failure.
  llvm::Expected<void (*)(void **)> lookup(llvm::StringRef name) const;

  /// Invokes the function with the given name passing it the list of opaque
  /// pointers to the actual arguments.
  llvm::Error
  invokePacked(llvm::StringRef name,
               llvm::MutableArrayRef<void *> args = llvm::None) const;

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
    FnResult(T &result) : value(result) {}
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
  struct Argument<Result<T>> {
    static void pack(llvm::SmallVectorImpl<void *> &args, FnResult<T> &result) {
      args.push_back(&result.value);
    }
  };

  /// Invokes the function with the given name passing it the list of arguments
  /// by value. Function result can be obtain through output parameter using the
  /// `FnResult` wrapper defined above. For example:
  ///
  ///     func @foo(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface }
  ///
  /// can be invoked:
  ///
  ///     int32_t result = 0;
  ///     llvm::Error error = jit->invoke("foo", 42,
  ///                                     result(result));
  template <typename... Args>
  llvm::Error invoke(llvm::StringRef funcName, Args... args) {
    const std::string adapterName = std::string("") + funcName.str();
    llvm::SmallVector<void *> argsArray;
    // Pack every arguments in an array of pointers. Delegate the packing to a
    // trait so that it can be overridden per argument type.
    // TODO: replace with a fold expression when migrating to C++17.
    int dummy[] = {0, ((void)Argument<Args>::pack(argsArray, args), 0)...};
    (void)dummy;
    return invokePacked(adapterName, argsArray);
  };

  /// Dump object code to output file `filename`.
  void dumpToObjectFile(llvm::StringRef filename);

  /// Register symbols with this ExecutionEngine.
  void registerSymbols(
      llvm::function_ref<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)>
          symbolMap);

  std::unique_ptr<exprs::Expression> eval(SereneContext &ctx,
                                          std::string input);
};
} // namespace serene

#endif
