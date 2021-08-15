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
};
} // namespace serene

#endif
