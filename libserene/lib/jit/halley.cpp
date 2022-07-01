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

#include "serene/jit/halley.h"

#include "serene/context.h"     // for Seren...
#include "serene/options.h"     // for Options
#include "serene/types/types.h" // for Names...

#include <system_error> // for error...

#include <llvm/ADT/StringMapEntry.h> // for Strin...
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Triple.h>                                   // for Triple
#include <llvm/ADT/iterator.h>                                 // for itera...
#include <llvm/ExecutionEngine/JITEventListener.h>             // for JITEv...
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>             // for TMOwn...
#include <llvm/ExecutionEngine/Orc/Core.h>                     // for Execu...
#include <llvm/ExecutionEngine/Orc/DebugUtils.h>               // for opera...
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>           // for Dynam...
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>           // for IRCom...
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>  // for JITTa...
#include <llvm/ExecutionEngine/Orc/LLJIT.h>                    // for LLJIT...
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h> // for RTDyl...
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>         // for Threa...
#include <llvm/ExecutionEngine/SectionMemoryManager.h>         // for Secti...
#include <llvm/IR/DataLayout.h>                                // for DataL...
#include <llvm/IR/LLVMContext.h>                               // for LLVMC...
#include <llvm/IR/Module.h>                                    // for Module
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/CodeGen.h> // for Level
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>     // for OF_None
#include <llvm/Support/FormatVariadic.h> // for formatv
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h> // for ToolO...
#include <llvm/Support/raw_ostream.h>    // for raw_o...

#include <algorithm> // for max
#include <assert.h>  // for assert
#include <cerrno>
#include <cstring>
#include <gc.h>
#include <memory>  // for uniqu...
#include <string>  // for opera...
#include <utility> // for move

#define COMMON_ARGS_COUNT 8

namespace serene {

namespace jit {

void ObjectCache::notifyObjectCompiled(const llvm::Module *m,
                                       llvm::MemoryBufferRef objBuffer) {
  cachedObjects[m->getModuleIdentifier()] =
      llvm::MemoryBuffer::getMemBufferCopy(objBuffer.getBuffer(),
                                           objBuffer.getBufferIdentifier());
}

std::unique_ptr<llvm::MemoryBuffer>
ObjectCache::getObject(const llvm::Module *m) {
  auto i = cachedObjects.find(m->getModuleIdentifier());

  if (i == cachedObjects.end()) {
    HALLEY_LOG("No object for " + m->getModuleIdentifier() +
               " in cache. Compiling.");
    return nullptr;
  }
  HALLEY_LOG("Object for " + m->getModuleIdentifier() + " loaded from cache.");
  return llvm::MemoryBuffer::getMemBuffer(i->second->getMemBufferRef());
}

void ObjectCache::dumpToObjectFile(llvm::StringRef outputFilename) {
  // Set up the output file.
  std::error_code error;

  auto file = std::make_unique<llvm::ToolOutputFile>(outputFilename, error,
                                                     llvm::sys::fs::OF_None);
  if (error) {

    llvm::errs() << "cannot open output file '" + outputFilename.str() +
                        "': " + error.message()
                 << "\n";
    return;
  }
  // Dump the object generated for a single module to the output file.
  // TODO: Replace this with a runtime check
  assert(cachedObjects.size() == 1 && "Expected only one object entry.");

  auto &cachedObject = cachedObjects.begin()->second;
  file->os() << cachedObject->getBuffer();
  file->keep();
}

llvm::orc::JITDylib *Halley::getLatestJITDylib(types::Namespace &ns) {

  if (jitDylibs.count(ns.name->data) == 0) {
    return nullptr;
  }

  auto vec = jitDylibs[ns.name->data];
  // TODO: Make sure that the returning Dylib still exists in the JIT
  //       by calling jit->engine->getJITDylibByName(dylib_name);
  return vec.empty() ? nullptr : vec.back();
};

void Halley::pushJITDylib(types::Namespace &ns, llvm::orc::JITDylib *l) {
  if (jitDylibs.count(ns.name->data) == 0) {
    llvm::SmallVector<llvm::orc::JITDylib *, 1> vec{l};
    jitDylibs[ns.name->data] = vec;
    return;
  }
  auto vec = jitDylibs[ns.name->data];
  vec.push_back(l);
  jitDylibs[ns.name->data] = vec;
}

size_t Halley::getNumberOfJITDylibs(types::Namespace &ns) {
  if (jitDylibs.count(ns.name->data) == 0) {
    return 0;
  }

  return jitDylibs[ns.name->data].size();
};

Halley::Halley(std::unique_ptr<SereneContext> ctx,
               llvm::orc::JITTargetMachineBuilder &&jtmb, llvm::DataLayout &&dl)
    : cache(ctx->opts.JITenableObjectCache ? new ObjectCache() : nullptr),
      gdbListener(ctx->opts.JITenableGDBNotificationListener

                      ? llvm::JITEventListener::createGDBRegistrationListener()
                      : nullptr),
      perfListener(ctx->opts.JITenablePerfNotificationListener
                       ? llvm::JITEventListener::createPerfJITEventListener()
                       : nullptr),
      jtmb(jtmb), dl(dl), ctx(std::move(ctx)){};

// MaybeJITPtr Halley::lookup(exprs::Symbol &sym) const {
//   HALLEY_LOG("Looking up: " << sym.toString());
//   auto *ns = ctx.getNS(sym.nsName);

//   if (ns == nullptr) {
//     return errors::makeError(ctx, errors::CantResolveSymbol, sym.location,
//                              "Can't find the namespace in the context: " +
//                                  sym.nsName);
//   }

//   auto *dylib = ctx.getLatestJITDylib(*ns);
//   //

//   if (dylib == nullptr) {
//     return errors::makeError(ctx, errors::CantResolveSymbol, sym.location,
//                              "Don't know about namespace: " + sym.nsName);
//   }

//   auto expectedSymbol =
//       engine->lookup(*dylib, makePackedFunctionName(sym.name));

//   // JIT lookup may return an Error referring to strings stored internally by
//   // the JIT. If the Error outlives the ExecutionEngine, it would want have a
//   // dangling reference, which is currently caught by an assertion inside JIT
//   // thanks to hand-rolled reference counting. Rewrap the error message into
//   a
//   // string before returning. Alternatively, ORC JIT should consider copying
//   // the string into the error message.
//   if (!expectedSymbol) {
//     std::string errorMessage;
//     llvm::raw_string_ostream os(errorMessage);
//     llvm::handleAllErrors(expectedSymbol.takeError(),
//                           [&os](llvm::ErrorInfoBase &ei) { ei.log(os); });
//     return errors::makeError(ctx, errors::CantResolveSymbol, sym.location,
//                              os.str());
//   }

//   auto rawFPtr = expectedSymbol->getValue();
//   // NOLINTNEXTLINE(performance-no-int-to-ptr)
//   auto fptr = reinterpret_cast<void (*)(void **)>(rawFPtr);

//   if (fptr == nullptr) {
//     return errors::makeError(ctx, errors::CantResolveSymbol, sym.location,
//                              "Lookup function is null!");
//   }

//   return fptr;
// };

void Halley::setEngine(std::unique_ptr<llvm::orc::LLJIT> e, bool isLazy) {
  // Later on we might use different classes of JIT which might need some
  // work for lazyness
  (void)ctx;
  engine       = std::move(e);
  this->isLazy = isLazy;
};

void Halley::dumpToObjectFile(llvm::StringRef filename) {
  cache->dumpToObjectFile(filename);
};

MaybeEngine Halley::make(std::unique_ptr<SereneContext> sereneCtxPtr,
                         llvm::orc::JITTargetMachineBuilder &&jtmb) {
  auto dl = jtmb.getDefaultDataLayoutForTarget();
  if (!dl) {
    return dl.takeError();
  }

  auto jitEngine = std::make_unique<Halley>(std::move(sereneCtxPtr),
                                            std::move(jtmb), std::move(*dl));

  // Why not the llvmcontext from the SereneContext??
  // Sice we're going to pass the ownership of this context to a thread
  // safe module later on and we will only create the jit function wrappers
  // with it, then it is fine to use a new context.
  //
  // What might go wrong?
  // in a repl env when we have to create new modules on top of each other
  // having two different contex might be a problem, but i think since we
  // use the first context to generate the IR and the second one to just
  // run it.
  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext);

  // Since we moved the original sereneCtxPtr into the engine.
  auto &sereneCtx = jitEngine->getContext();

  // Callback to create the object layer with symbol resolution to current
  // process and dynamically linked libraries.
  auto objectLinkingLayerCreator = [&](llvm::orc::ExecutionSession &session,
                                       const llvm::Triple &tt) {
    (void)tt;

    auto objectLayer =
        std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(session, []() {
          return std::make_unique<llvm::SectionMemoryManager>();
        });

    // Register JIT event listeners if they are enabled.
    if (jitEngine->gdbListener != nullptr) {
      objectLayer->registerJITEventListener(*jitEngine->gdbListener);
    }
    if (jitEngine->perfListener != nullptr) {
      objectLayer->registerJITEventListener(*jitEngine->perfListener);
    }

    // COFF format binaries (Windows) need special handling to deal with
    // exported symbol visibility.
    // cf llvm/lib/ExecutionEngine/Orc/LLJIT.cpp
    // LLJIT::createObjectLinkingLayer

    if (sereneCtx.triple.isOSBinFormatCOFF()) {
      objectLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      objectLayer->setAutoClaimResponsibilityForObjectSymbols(true);
    }

    // Resolve symbols from shared libraries.
    // for (auto libPath : sharedLibPaths) {
    //   auto mb = llvm::MemoryBuffer::getFile(libPath);
    //   if (!mb) {
    //     llvm::errs() << "Failed to create MemoryBuffer for: " << libPath
    //                  << "\nError: " << mb.getError().message() << "\n";
    //     continue;
    //   }
    //   auto &JD    = session.createBareJITDylib(std::string(libPath));
    //   auto loaded = llvm::orc::DynamicLibrarySearchGenerator::Load(
    //       libPath.data(), dataLayout.getGlobalPrefix());
    //   if (!loaded) {
    //     llvm::errs() << "Could not load " << libPath << ":\n  "
    //                  << loaded.takeError() << "\n";
    //     continue;
    //   }

    //   JD.addGenerator(std::move(*loaded));
    //   cantFail(objectLayer->add(JD, std::move(mb.get())));
    // }

    return objectLayer;
  };

  // Callback to inspect the cache and recompile on demand. This follows Lang's
  // LLJITWithObjectCache example.
  auto compileFunctionCreator = [&](llvm::orc::JITTargetMachineBuilder JTMB)
      -> llvm::Expected<
          std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
    llvm::CodeGenOpt::Level jitCodeGenOptLevel =
        static_cast<llvm::CodeGenOpt::Level>(sereneCtx.getOptimizatioLevel());

    JTMB.setCodeGenOptLevel(jitCodeGenOptLevel);

    auto targetMachine = JTMB.createTargetMachine();
    if (!targetMachine) {
      return targetMachine.takeError();
    }

    return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(
        std::move(*targetMachine), jitEngine->cache.get());
  };

  if (sereneCtx.opts.JITLazy) {
    // Setup a LLLazyJIT instance to the times that latency is important
    // for example in a REPL. This way

    auto jit =
        cantFail(llvm::orc::LLLazyJITBuilder()
                     .setCompileFunctionCreator(compileFunctionCreator)
                     .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
                     .create());
    jitEngine->setEngine(std::move(jit), true);

  } else {
    // Setup a LLJIT instance for the times that performance is important
    // and we want to compile everything as soon as possible. For instance
    // when we run the JIT in the compiler
    auto jit =
        cantFail(llvm::orc::LLJITBuilder()
                     .setCompileFunctionCreator(compileFunctionCreator)
                     .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
                     .create());

    jitEngine->setEngine(std::move(jit), false);
  }

  jitEngine->engine->getIRCompileLayer().setNotifyCompiled(
      [&](llvm::orc::MaterializationResponsibility &r,
          llvm::orc::ThreadSafeModule tsm) {
        auto syms = r.getRequestedSymbols();
        tsm.withModuleDo([&](llvm::Module &m) {
          HALLEY_LOG("Compiled "
                     << syms << " for the module: " << m.getModuleIdentifier());
        });
      });

  // Resolve symbols that are statically linked in the current process.
  llvm::orc::JITDylib &mainJD = jitEngine->engine->getMainJITDylib();
  mainJD.addGenerator(
      cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          jitEngine->dl.getGlobalPrefix())));

  return MaybeEngine(std::move(jitEngine));
};

const types::InternalString &Halley::getInternalString(const char *s) {
  // TODO: [serene.core] We need to provide some functions on llvm level to
  // build instances from these type in a functional way. We need to avoid
  // randomly build instances here and there that causes unsafe memory
  assert(s && "s is nullptr: getInternalString");
  auto len = std::strlen(s);

  auto *str =
      (types::InternalString *)GC_MALLOC_ATOMIC(sizeof(types::InternalString));

  str->data = (char *)GC_MALLOC_ATOMIC(len);
  memcpy((void *)str->data, (void *)s, len);
  str->len = len;

  stringStorage.push_back(str);
  return *str;
  // /TODO
};

types::Namespace &Halley::createNamespace(const char *name) {
  // TODO: [serene.core] We need to provide some functions on llvm level to
  // build instances from these type in a functional way. We need to avoid
  // randomly build instances here and there that causes unsafe memory
  assert(name && "name is nullptr: createNamespace");
  const auto &nsName = getInternalString(name);
  auto *ns           = (types::Namespace *)GC_MALLOC(sizeof(types::Namespace));
  ns->name           = &nsName;

  nsStorage.push_back(ns);
  return *ns;
  // /TODO
};

llvm::Error Halley::createEmptyNS(const char *name) {
  assert(name && "name is nullptr: createEmptyNS");
  auto &ns         = createNamespace(name);
  auto numOfDylibs = getNumberOfJITDylibs(ns) + 1;

  HALLEY_LOG(
      llvm::formatv("Creating Dylib {0}#{1}", ns.name->data, numOfDylibs));

  auto newDylib = engine->createJITDylib(
      llvm::formatv("{0}#{1}", ns.name->data, numOfDylibs));

  if (!newDylib) {
    llvm::errs() << "Couldn't create the jitDylib\n";
    serene::terminate(*ctx, 1);
  }

  pushJITDylib(ns, &(*newDylib));
  return llvm::Error::success();
};

llvm::Error Halley::loadModule(const char *file) {
  assert(file && "File is nullptr: loadModule");
  auto llvmContext = ctx->genLLVMContext();
  llvm::SMDiagnostic error;

  auto module = llvm::parseIRFile(file, error, *llvmContext);

  if (module == nullptr) {
    return llvm::make_error<llvm::StringError>(
        std::make_error_code(std::errc::executable_format_error),
        error.getMessage().str() + " File: " + file);
  }

  return llvm::Error::success();
};

MaybeEngine makeHalleyJIT(std::unique_ptr<SereneContext> ctx) {
  llvm::orc::JITTargetMachineBuilder jtmb(ctx->triple);
  auto maybeJIT = Halley::make(std::move(ctx), std::move(jtmb));
  if (!maybeJIT) {
    return maybeJIT.takeError();
  }

  return maybeJIT;
};

} // namespace jit
} // namespace serene
