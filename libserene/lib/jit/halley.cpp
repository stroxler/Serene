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

#include "serene/context.h" // for Seren...
#include "serene/fs.h"
#include "serene/options.h"     // for Options
#include "serene/types/types.h" // for Names...

#include <system_error> // for error...

#include <llvm/ADT/StringMapEntry.h> // for Strin...
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Triple.h>   // for Triple
#include <llvm/ADT/iterator.h> // for itera...
#include <llvm/BinaryFormat/Magic.h>
#include <llvm/ExecutionEngine/JITEventListener.h> // for JITEv...
#include <llvm/ExecutionEngine/Orc/CompileUtils.h> // for TMOwn...
#include <llvm/ExecutionEngine/Orc/Core.h>         // for Execu...
#include <llvm/ExecutionEngine/Orc/DebugUtils.h>   // for opera...
#include <llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>          // for Dynam...
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>          // for IRCom...
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h> // for JITTa...
#include <llvm/ExecutionEngine/Orc/LLJIT.h>                   // for LLJIT...
#include <llvm/ExecutionEngine/Orc/ObjectFileInterface.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
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

// TODO: [error] Replace this function when we implemented
// the error subsystem with the official implementation
llvm::Error tempError(SereneContext &ctx, llvm::Twine s) {
  (void)ctx;
  return llvm::make_error<llvm::StringError>(
      std::make_error_code(std::errc::executable_format_error),
      "[Error]: " + s);
};
// /TODO

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

llvm::orc::JITDylib *Halley::getLatestJITDylib(const types::Namespace &ns) {
  return getLatestJITDylib(ns.name->data);
};

llvm::orc::JITDylib *Halley::getLatestJITDylib(const char *nsName) {
  if (jitDylibs.count(nsName) == 0) {
    return nullptr;
  }

  auto vec = jitDylibs[nsName];
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

    return objectLayer;
  };

  // Callback to inspect the cache and recompile on demand. This follows
  // Lang's LLJITWithObjectCache example.
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

  // TODO: [jit] This is not a proper way to handle both engines.
  // Create two different classes for different execution modes
  // (`lazy` vs `eager`) with the same interface and use them
  // where appropriate.
  if (sereneCtx.opts.JITLazy) {
    // Setup a LLLazyJIT instance to the times that latency is important
    // for example in a REPL. This way

    auto jit =
        cantFail(llvm::orc::LLLazyJITBuilder()
                     .setCompileFunctionCreator(compileFunctionCreator)
                     .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
                     .create());

    jitEngine->setEngine(std::move(jit), sereneCtx.opts.JITLazy);
  } else {
    // Setup a LLJIT instance for the times that performance is important
    // and we want to compile everything as soon as possible. For instance
    // when we run the JIT in the compiler
    auto jit =
        cantFail(llvm::orc::LLJITBuilder()
                     .setCompileFunctionCreator(compileFunctionCreator)
                     .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
                     .create());
    jitEngine->setEngine(std::move(jit), sereneCtx.opts.JITLazy);
  }
  // /TODO

  jitEngine->engine->getIRCompileLayer().setNotifyCompiled(
      [&](llvm::orc::MaterializationResponsibility &r,
          llvm::orc::ThreadSafeModule tsm) {
        auto syms = r.getRequestedSymbols();
        tsm.withModuleDo([&](llvm::Module &m) {
          HALLEY_LOG("Compiled "
                     << syms << " for the module: " << m.getModuleIdentifier());
        });
      });

  if (auto err = jitEngine->createCurrentProcessJD()) {
    return err;
  }

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

types::Namespace &Halley::makeNamespace(const char *name) {
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
  // TODO: Look up the Namespace first.
  auto &ns         = makeNamespace(name);
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

MaybeJitAddress Halley::lookup(const types::Symbol &sym) const {
  return lookup(sym.ns->data, sym.name->data);
}

MaybeJitAddress Halley::lookup(const char *nsName, const char *sym) const {
  assert(sym != nullptr && "'sym' is null: lookup");
  assert(nsName != nullptr && "'nsName' is null: lookup");

  llvm::StringRef s{sym};
  llvm::StringRef ns{nsName};

  std::string fqsym = (ns + "/" + s).str();

  HALLEY_LOG("Looking up symbol: " << fqsym);
  auto *dylib = const_cast<Halley *>(this)->jitDylibs[nsName].back();

  if (dylib == nullptr) {
    return tempError(*ctx, "No dylib " + s);
  }

  HALLEY_LOG("Looking in dylib: " << (void *)dylib);
  auto expectedSymbol = engine->lookup(*dylib, fqsym);

  // JIT lookup may return an Error referring to strings stored internally by
  // the JIT. If the Error outlives the ExecutionEngine, it would want have a
  // dangling reference, which is currently caught by an assertion inside JIT
  // thanks to hand-rolled reference counting. Rewrap the error message into a
  // string before returning. Alternatively, ORC JIT should consider copying
  // the string into the error message.
  if (!expectedSymbol) {
    return expectedSymbol.takeError();
  }

  auto rawFPtr = expectedSymbol->getAddress();

  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto fptr = reinterpret_cast<JitWrappedAddress>(rawFPtr);

  if (fptr == nullptr) {
    return tempError(*ctx, "Lookup function is null!");
  }

  HALLEY_LOG("Found symbol '" << fqsym << "' at " << (void *)fptr);
  return fptr;
};

// TODO: Remove this function before prod release
llvm::Error Halley::loadModule(const char *nsName, const char *file) {
  assert(file && "'file' is nullptr: loadModule");
  assert(nsName && "'nsName' is nullptr: loadModule");

  auto llvmContext = ctx->genLLVMContext();
  llvm::SMDiagnostic error;

  auto *dylib = jitDylibs[nsName].back();

  auto module = llvm::parseIRFile(file, error, *llvmContext);

  if (module == nullptr) {
    return llvm::make_error<llvm::StringError>(
        std::make_error_code(std::errc::executable_format_error),
        error.getMessage().str() + " File: " + file);
  }

  auto tsm =
      llvm::orc::ThreadSafeModule(std::move(module), std::move(llvmContext));

  return engine->addIRModule(*dylib, std::move(tsm));
};
// /TODO

// TODO: [error] Remove this function when we implemented
// the error subsystem
llvm::Error NotImplemented(llvm::StringRef s) {
  return llvm::make_error<llvm::StringError>(
      std::make_error_code(std::errc::executable_format_error),
      "Not Implemented: " + s);
};

template <>
MaybeDylibPtr
Halley::loadNamespaceFrom<fs::NSFileType::Source>(NSLoadRequest &req) {
  (void)req;
  return nullptr;
};

template <>
MaybeDylibPtr
Halley::loadNamespaceFrom<fs::NSFileType::TextIR>(NSLoadRequest &req) {
  (void)req;
  return nullptr;
};

template <>
MaybeDylibPtr
Halley::loadNamespaceFrom<fs::NSFileType::BinaryIR>(NSLoadRequest &req) {
  (void)req;
  return nullptr;
};

template <>
MaybeDylibPtr
Halley::loadNamespaceFrom<fs::NSFileType::ObjectFile>(NSLoadRequest &req) {

  auto file = fs::join(req.path, req.nsToFileName + ".o");

  if (!fs::exists(file)) {
    // Can't locate any object file, skit to the next loader
    HALLEY_LOG("File does not exist: " << file << "\n");
    return nullptr;
  }

  auto err = createEmptyNS(req.nsName.str().c_str());
  if (err) {
    return err;
  }

  auto *jd = getLatestJITDylib(req.nsName.str().c_str());
  assert(jd != nullptr && "'jd' must not be null since we just created it.");

  auto buf = llvm::errorOrToExpected(llvm::MemoryBuffer::getFile(file));
  if (!buf) {
    return buf.takeError();
  }

  err = engine->getObjLinkingLayer().add(*jd, std::move(*buf));

  if (err) {
    return err;
  }

  return jd;
};

template <>
MaybeDylibPtr
Halley::loadNamespaceFrom<fs::NSFileType::StaticLib>(NSLoadRequest &req) {

  // Skip missing or non-regular paths.
  if (llvm::sys::fs::get_file_type(req.path) !=
      llvm::sys::fs::file_type::regular_file) {
    return tempError(*ctx, "Not a regular file: " + req.path);
  }

  llvm::file_magic magic;
  if (auto ec = llvm::identify_magic(req.path, magic)) {
    // If there was an error loading the file then skip it.
    return tempError(*ctx,
                     ec.message() + "\nFile Identification Erorr: " + req.path);
  }

  if (magic != llvm::file_magic::archive ||
      magic != llvm::file_magic::macho_universal_binary) {
    return tempError(*ctx, "Not a static lib: " + req.path);
  }

  auto err = createEmptyNS(req.nsName.str().c_str());
  if (err) {
    return err;
  }

  auto &session = engine->getExecutionSession();
  auto *jd      = getLatestJITDylib(req.nsName.str().c_str());
  assert(jd == nullptr && "'jd' must not be null since we just created it.");

  // TODO: Handle hidden static libs as well look at the addLibrary/AddArchive
  // in llvm-jitlink

  auto generator = llvm::orc::StaticLibraryDefinitionGenerator::Load(
      engine->getObjLinkingLayer(), req.path.str().c_str(),
      session.getExecutorProcessControl().getTargetTriple(),
      std::move(llvm::orc::getObjectFileInterface));

  if (!generator) {
    return generator.takeError();
  }

  jd->addGenerator(std::move(*generator));
  return jd;
};

template <>
MaybeDylibPtr
Halley::loadNamespaceFrom<fs::NSFileType::SharedLib>(NSLoadRequest &req) {
  (void)req;
  //   switch (magic) {
  // case llvm::file_magic::elf_shared_object:
  // case llvm::file_magic::macho_dynamically_linked_shared_lib: {
  //   // TODO: On first reference to LibPath this should create a JITDylib
  //   // with a generator and add it to JD's links-against list. Subsquent
  //   // references should use the JITDylib created on the first
  //   // reference.
  //   auto g = llvm::EPCDynamicLibrarySearchGenerator::Load(session, path);
  //   if (!g)
  //     return g.takeError();
  //   jd.addGenerator(std::move(*g));
  //   break;
  // }
  // case llvm::file_magic::archive:
  // case llvm::file_magic::macho_universal_binary: {
  // }
  // default:
  //   // This file isn't a recognized library kind.
  //   LibFound = false;
  //   break;
  // }

  return NotImplemented("loadNamespaceFrom<shared>");
};

MaybeDylibPtr Halley::loadNamespaceFrom(fs::NSFileType type_,
                                        NSLoadRequest &req) {
  switch (type_) {
  case fs::NSFileType::Source:
    return loadNamespaceFrom<fs::NSFileType::Source>(req);
  case fs::NSFileType::TextIR:
    return loadNamespaceFrom<fs::NSFileType::TextIR>(req);
  case fs::NSFileType::BinaryIR:
    return loadNamespaceFrom<fs::NSFileType::BinaryIR>(req);
  case fs::NSFileType::ObjectFile:
    return loadNamespaceFrom<fs::NSFileType::ObjectFile>(req);
  case fs::NSFileType::StaticLib:
  case fs::NSFileType::SharedLib:
    return loadNamespaceFrom<fs::NSFileType::StaticLib>(req);
  };
};

MaybeDylibPtr Halley::loadNamespace(std::string &nsName) {
  if (ctx->getLoadPaths().empty()) {
    return tempError(*ctx, "Load paths should not be empty");
  }

  for (auto &path : ctx->getLoadPaths()) {
    auto nsFileName = fs::namespaceToPath(nsName);
    NSLoadRequest req{nsName, path, nsFileName};

    for (auto type_ : {fs::NSFileType::Source, fs::NSFileType::ObjectFile}) {
      auto maybeJDptr = loadNamespaceFrom(type_, req);

      if (!maybeJDptr) {
        return maybeJDptr.takeError();
      }

      if (*maybeJDptr != nullptr) {
        auto *processJD = engine->getExecutionSession().getJITDylibByName(
            MAIN_PROCESS_JD_NAME);

        if (processJD == nullptr) {
          // TODO: [jit] Panic here
          return tempError(*ctx, "Can't find the main process JD");
          // /TODO
        }

        (*maybeJDptr)->addToLinkOrder(*processJD);
        return *maybeJDptr;
      }
    }
  }

  return tempError(*ctx, "Can't find namespace: " + nsName);
};

MaybeDylibPtr Halley::loadStaticLibrary(const std::string &name) {
  if (ctx->getLoadPaths().empty()) {
    return tempError(*ctx, "Load paths should not be empty");
  }

  for (auto &path : ctx->getLoadPaths()) {
    auto file = fs::join(path, name + ".a");

    if (!fs::exists(file)) {
      continue;
    }

    if (!fs::isStaticLib(file)) {
      return tempError(*ctx, "Not a static lib: " + file);
    }

    auto *objectLayer = &engine->getObjLinkingLayer();

    auto generator = llvm::orc::StaticLibraryDefinitionGenerator::Load(
        *objectLayer, file.c_str(),
        engine->getExecutionSession()
            .getExecutorProcessControl()
            .getTargetTriple(),
        std::move(llvm::orc::getObjectFileInterface));

    if (!generator) {
      return generator.takeError();
    }

    auto jd = engine->createJITDylib(llvm::formatv("{0}#{1}", name, 0));

    if (!jd) {
      return jd.takeError();
    }

    jd->addGenerator(std::move(*generator));

    std::vector<llvm::StringRef> nsNames = {name};

    auto definition = engine->lookup(*jd, "__serene_namespaces");

    if (!definition) {
      HALLEY_LOG("Library '" << name << "' is not a Serene lib.");
      // We just want to ignore the error
      llvm::consumeError(definition.takeError());
    } else {
      HALLEY_LOG("Library '" << name << "' is a Serene lib.");
      // TODO: call the __serene_namespaces and set nsNames to
      // the list of namespaces that it returns
      (void)*definition;
    }

    for (auto &nsName : nsNames) {
      auto ns = makeNamespace(nsName.str().c_str());
      pushJITDylib(ns, &(*jd));
    }

    return &jd.get();
  }

  return tempError(*ctx, "Can't find static lib: " + name);
};

MaybeDylibPtr Halley::loadSharedLibFile(llvm::StringRef name,
                                        llvm::StringRef path) {
  if (!fs::isSharedLib(path)) {
    return tempError(*ctx, "Not a shared lib: " + path);
  }

  auto generator = llvm::orc::EPCDynamicLibrarySearchGenerator::Load(
      engine->getExecutionSession(), path.str().c_str());

  if (!generator) {
    return generator.takeError();
  }

  auto jd = engine->createJITDylib(llvm::formatv("{0}#{1}", name, 0));

  if (!jd) {
    return jd.takeError();
  }

  jd->addGenerator(std::move(*generator));

  return &jd.get();
};

MaybeDylibPtr Halley::loadSharedLibrary(const std::string &name) {
  if (ctx->getLoadPaths().empty()) {
    return tempError(*ctx, "Load paths should not be empty");
  }

  for (auto &path : ctx->getLoadPaths()) {
    auto file = fs::join(path, name + ".so");

    if (!fs::exists(file)) {
      continue;
    }

    auto maybeJD = loadSharedLibFile(name, file);
    if (!maybeJD) {
      return maybeJD.takeError();
    }

    auto *jd     = *maybeJD;
    auto nsNames = getContainedNamespaces(name, jd);

    for (const auto *nsName : nsNames) {
      auto ns = makeNamespace(nsName);
      pushJITDylib(ns, jd);
    }

    return jd;
  }

  return tempError(*ctx, "Can't find the dynamic lib: " + name);
};

std::vector<const char *> Halley::getContainedNamespaces(llvm::StringRef name,
                                                         DylibPtr jd) {

  std::vector<const char *> nsNames = {name.str().c_str()};
  auto definition = engine->lookup(*jd, "__serene_namespaces");

  if (!definition) {
    HALLEY_LOG("Library is not a Serene lib.");
    // We just want to ignore the error
    llvm::consumeError(definition.takeError());
  }
  HALLEY_LOG("Library is a Serene lib.");
  // TODO: call the __serene_namespaces and set nsNames to
  // the list of namespaces that it returns

  return nsNames;
};

llvm::Error Halley::createCurrentProcessJD() {

  auto &es           = engine->getExecutionSession();
  auto *processJDPtr = es.getJITDylibByName(MAIN_PROCESS_JD_NAME);

  if (processJDPtr != nullptr) {
    // We already created the JITDylib for the current process
    return llvm::Error::success();
  }

  auto processJD = es.createJITDylib(MAIN_PROCESS_JD_NAME);

  if (!processJD) {
    return processJD.takeError();
  }

  auto generator =
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          engine->getDataLayout().getGlobalPrefix());

  if (!generator) {
    return generator.takeError();
  }

  processJD->addGenerator(std::move(*generator));
  return llvm::Error::success();
};

llvm::Error Halley::invokePacked(const types::Symbol &name,
                                 llvm::MutableArrayRef<void *> args) const {
  auto expectedFPtr = lookup(name);
  if (!expectedFPtr) {
    return expectedFPtr.takeError();
  }

  auto *fptr = *expectedFPtr;
  (*fptr)(args.data());

  return llvm::Error::success();
}

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
