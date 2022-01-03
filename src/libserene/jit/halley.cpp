/*
 * Serene programming language.
 *
 *  Copyright (c) 2020 Sameer Rahmani <lxsameer@gnu.org>
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
 *
 * Most of the codes in this file is borrowed from the LLVM project so the LLVM
 * license applies here
 */

#include "serene/jit/halley.h"

#include "serene/context.h"
#include "serene/diagnostics.h"
#include "serene/errors/constants.h"
#include "serene/errors/error.h"
#include "serene/exprs/symbol.h"
#include "serene/namespace.h"
#include "serene/utils.h"

#include <llvm-c/Types.h>
#include <llvm/ADT/None.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/IRTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/FormatVariadicDetails.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/Support/FileUtilities.h>

#include <memory>
#include <stdexcept>
#include <vector>

#define COMMON_ARGS_COUNT 8
// Just to make the linter happy
#define I64_BIT_SIZE 64

namespace serene {

namespace jit {
// TODO: Remove this function and replace it by our own version of
//       error handler
/// Wrap a string into an llvm::StringError.
// static llvm::Error make_string_error(const llvm::Twine &message) {
//   return llvm::make_error<llvm::StringError>(message.str(),
//                                              llvm::inconvertibleErrorCode());
// }

static std::string makePackedFunctionName(llvm::StringRef name) {
  return "_serene_" + name.str();
}

static void packFunctionArguments(llvm::Module *module) {
  auto &ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  llvm::DenseSet<llvm::Function *> interfaceFunctions;
  for (auto &func : module->getFunctionList()) {
    if (func.isDeclaration()) {
      continue;
    }
    if (interfaceFunctions.count(&func) != 0) {
      continue;
    }

    // Given a function `foo(<...>)`, define the interface function
    // `serene_foo(i8**)`.
    auto *newType = llvm::FunctionType::get(
        builder.getVoidTy(), builder.getInt8PtrTy()->getPointerTo(),
        /*isVarArg=*/false);
    auto newName = makePackedFunctionName(func.getName());
    auto funcCst = module->getOrInsertFunction(newName, newType);
    llvm::Function *interfaceFunc =
        llvm::cast<llvm::Function>(funcCst.getCallee());
    interfaceFunctions.insert(interfaceFunc);

    // Extract the arguments from the type-erased argument list and cast them to
    // the proper types.
    auto *bb = llvm::BasicBlock::Create(ctx);
    bb->insertInto(interfaceFunc);
    builder.SetInsertPoint(bb);
    llvm::Value *argList = interfaceFunc->arg_begin();
    llvm::SmallVector<llvm::Value *, COMMON_ARGS_COUNT> args;
    args.reserve(llvm::size(func.args()));
    for (const auto &indexedArg : llvm::enumerate(func.args())) {
      llvm::Value *argIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), llvm::APInt(I64_BIT_SIZE, indexedArg.index()));
      llvm::Value *argPtrPtr =
          builder.CreateGEP(builder.getInt8PtrTy(), argList, argIndex);
      llvm::Value *argPtr =
          builder.CreateLoad(builder.getInt8PtrTy(), argPtrPtr);
      llvm::Type *argTy = indexedArg.value().getType();
      argPtr            = builder.CreateBitCast(argPtr, argTy->getPointerTo());
      llvm::Value *arg  = builder.CreateLoad(argTy, argPtr);
      args.push_back(arg);
    }

    // Call the implementation function with the extracted arguments.
    llvm::Value *result = builder.CreateCall(&func, args);

    // Assuming the result is one value, potentially of type `void`.
    if (!result->getType()->isVoidTy()) {
      llvm::Value *retIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(),
          llvm::APInt(I64_BIT_SIZE, llvm::size(func.args())));
      llvm::Value *retPtrPtr =
          builder.CreateGEP(builder.getInt8PtrTy(), argList, retIndex);
      llvm::Value *retPtr =
          builder.CreateLoad(builder.getInt8PtrTy(), retPtrPtr);
      retPtr = builder.CreateBitCast(retPtr, result->getType()->getPointerTo());
      builder.CreateStore(result, retPtr);
    }

    // The interface function returns void.
    builder.CreateRetVoid();
  }
};

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
               " in cache. Compiling.\n");
    return nullptr;
  }
  HALLEY_LOG("Object for " + m->getModuleIdentifier() +
             " loaded from cache.\n");
  return llvm::MemoryBuffer::getMemBuffer(i->second->getMemBufferRef());
}

void ObjectCache::dumpToObjectFile(llvm::StringRef outputFilename) {
  // Set up the output file.
  std::string errorMessage;
  auto file = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return;
  }

  // Dump the object generated for a single module to the output file.
  // TODO: Replace this with a runtime check
  assert(cachedObjects.size() == 1 && "Expected only one object entry.");

  auto &cachedObject = cachedObjects.begin()->second;
  file->os() << cachedObject->getBuffer();
  file->keep();
}

Halley::Halley(serene::SereneContext &ctx,
               llvm::orc::JITTargetMachineBuilder &&jtmb, llvm::DataLayout &&dl)
    : cache(ctx.opts.JITenableObjectCache ? new ObjectCache() : nullptr),
      gdbListener(ctx.opts.JITenableGDBNotificationListener

                      ? llvm::JITEventListener::createGDBRegistrationListener()
                      : nullptr),
      perfListener(ctx.opts.JITenablePerfNotificationListener
                       ? llvm::JITEventListener::createPerfJITEventListener()
                       : nullptr),
      jtmb(jtmb), dl(dl), ctx(ctx),
      activeNS(ctx.getSharedPtrToNS(ctx.getCurrentNS().name)) {
  assert(activeNS != nullptr && "Active NS is null!!!");
};

MaybeJITPtr Halley::lookup(exprs::Symbol &sym) const {
  HALLEY_LOG("Looking up: " << sym.toString());
  auto *ns = ctx.getNS(sym.nsName);

  if (ns == nullptr) {
    return MaybeJITPtr::error(errors::makeErrorTree(
        sym.location, errors::CantResolveSymbol,
        "Can't find the namespace in the context: " + sym.nsName));
  }

  auto *dylib = ctx.getLatestJITDylib(*ns);
  //

  if (dylib == nullptr) {
    return MaybeJITPtr::error(
        errors::makeErrorTree(sym.location, errors::CantResolveSymbol,
                              "Don't know about namespace: " + sym.nsName));
  }

  auto expectedSymbol =
      engine->lookup(*dylib, makePackedFunctionName(sym.name));
  // auto expectedSymbol = engine->lookup(name);
  //  JIT lookup may return an Error referring to strings stored internally by
  //  the JIT. If the Error outlives the ExecutionEngine, it would want have a
  //  dangling reference, which is currently caught by an assertion inside JIT
  //  thanks to hand-rolled reference counting. Rewrap the error message into a
  //  string before returning. Alternatively, ORC JIT should consider copying
  //  the string into the error message.
  if (!expectedSymbol) {
    std::string errorMessage;
    llvm::raw_string_ostream os(errorMessage);
    llvm::handleAllErrors(expectedSymbol.takeError(),
                          [&os](llvm::ErrorInfoBase &ei) { ei.log(os); });
    return MaybeJITPtr::error(errors::makeErrorTree(
        sym.location, errors::CantResolveSymbol, os.str()));
  }

  auto rawFPtr = expectedSymbol->getAddress();
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto fptr = reinterpret_cast<void (*)(void **)>(rawFPtr);

  if (fptr == nullptr) {
    return MaybeJITPtr::error(errors::makeErrorTree(
        sym.location, errors::CantResolveSymbol, "Lookup function is null!"));
  }

  return fptr;
};

void createObjectFile(SereneContext &ctx, llvm::StringRef name,
                      llvm::MemoryBufferRef objBuffer) {
  std::string errorMessage;
  auto file = mlir::openOutputFile(name, &errorMessage);
  if (!file) {

    panic(ctx, errorMessage);
  }

  file->os() << objBuffer.getBuffer();
  file->keep();
}

// llvm::Error Halley::invokePacked(llvm::StringRef name,
//                                  llvm::MutableArrayRef<void *> args) const {
//   auto expectedFPtr = lookup(name);
//   if (!expectedFPtr) {
//     return expectedFPtr.takeError();
//   }
//   auto fptr = *expectedFPtr;

//   (*fptr)(args.data());

//   return llvm::Error::success();
// }

void Halley::registerSymbols(
    llvm::function_ref<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)>
        symbolMap) {
  auto &mainJitDylib = engine->getMainJITDylib();
  cantFail(mainJitDylib.define(
      absoluteSymbols(symbolMap(llvm::orc::MangleAndInterner(
          mainJitDylib.getExecutionSession(), engine->getDataLayout())))));
};

llvm::Optional<errors::ErrorTree> Halley::addNS(Namespace &ns,
                                                reader::LocationRange &loc) {

  HALLEY_LOG(llvm::formatv("Creating Dylib {0}#{1}", ns.name,
                           ctx.getNumberOfJITDylibs(ns) + 1)
             << "\n");

  auto newDylib = engine->createJITDylib(
      llvm::formatv("{0}#{1}", ns.name, ctx.getNumberOfJITDylibs(ns) + 1));

  if (!newDylib) {
    return errors::makeErrorTree(loc, errors::CompilationError,
                                 "Filed to create dylib for " + ns.name);
  }

  ctx.pushJITDylib(ns, &(*newDylib));

  // TODO: Fix compileToLLVM to return proper errors
  auto maybeModule = ns.compileToLLVM();

  if (!maybeModule.hasValue()) {
    return errors::makeErrorTree(loc, errors::CompilationError);
  }

  auto tsm = std::move(maybeModule.getValue());
  tsm.withModuleDo([](llvm::Module &m) { packFunctionArguments(&m); });

  // TODO: Make sure that the data layout of the module is the same as the
  // engine
  cantFail(engine->addIRModule(*newDylib, std::move(tsm)));
  return llvm::None;
};

void Halley::setEngine(std::unique_ptr<llvm::orc::LLJIT> e, bool isLazy) {
  // Later on we might use different classes of JIT which might need some
  // work for lazyness
  engine       = std::move(e);
  this->isLazy = isLazy;
};

void Halley::dumpToObjectFile(llvm::StringRef filename) {
  cache->dumpToObjectFile(filename);
};

MaybeJIT Halley::make(SereneContext &serene_ctx,
                      llvm::orc::JITTargetMachineBuilder &&jtmb) {

  auto dl = jtmb.getDefaultDataLayoutForTarget();
  if (!dl) {
    return dl.takeError();
  }

  auto jitEngine =
      std::make_unique<Halley>(serene_ctx, std::move(jtmb), std::move(*dl));

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

  // Callback to create the object layer with symbol resolution to current
  // process and dynamically linked libraries.
  auto objectLinkingLayerCreator = [&](llvm::orc::ExecutionSession &session,
                                       const llvm::Triple &tt) {
    UNUSED(tt);

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
    llvm::Triple targetTriple(llvm::Twine(serene_ctx.targetTriple));

    if (targetTriple.isOSBinFormatCOFF()) {
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
        static_cast<llvm::CodeGenOpt::Level>(serene_ctx.getOptimizatioLevel());

    JTMB.setCodeGenOptLevel(jitCodeGenOptLevel);

    auto targetMachine = JTMB.createTargetMachine();
    if (!targetMachine) {
      return targetMachine.takeError();
    }

    return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(
        std::move(*targetMachine), jitEngine->cache.get());
  };

  if (serene_ctx.opts.JITLazy) {
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

  // Resolve symbols that are statically linked in the current process.
  llvm::orc::JITDylib &mainJD = jitEngine->engine->getMainJITDylib();
  mainJD.addGenerator(
      cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          jitEngine->dl.getGlobalPrefix())));

  return MaybeJIT(std::move(jitEngine));
};

Namespace &Halley::getActiveNS() { return *activeNS; };

llvm::Expected<std::unique_ptr<Halley>> makeHalleyJIT(SereneContext &ctx) {

  llvm::orc::JITTargetMachineBuilder jtmb(ctx.getTargetTriple());

  return Halley::make(ctx, std::move(jtmb));
};
} // namespace jit
} // namespace serene
