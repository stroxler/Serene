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

#ifndef SERENE_JIT_H
#define SERENE_JIT_H

#include "serene/errors.h"
#include "serene/slir/generatable.h"
#include "serene/utils.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/Debug.h>
#include <memory>
#include <mlir/Support/LLVM.h>

#define JIT_LOG(...) \
  DEBUG_WITH_TYPE("JIT", llvm::dbgs() << "[JIT]: " << __VA_ARGS__ << "\n");

namespace serene {
class JIT;

using MaybeJIT = Result<std::unique_ptr<JIT>, serene::errors::Error>;

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
  llvm::Error invokePacked(llvm::StringRef name,
                           llvm::MutableArrayRef<void *> args = llvm::None);

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
};
} // namespace serene

#endif
