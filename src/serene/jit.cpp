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

#include "serene/jit.h"

#include "serene/context.h"
#include "serene/namespace.h"

#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/IRTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/Support/FileUtilities.h>
#include <stdexcept>

namespace serene {

// TODO: Remove this function and replace it by our own version of
//       error handler
/// Wrap a string into an llvm::StringError.
static llvm::Error make_string_error(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

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
    if (interfaceFunctions.count(&func)) {
      continue;
    }

    // Given a function `foo(<...>)`, define the interface function
    // `mlir_foo(i8**)`.
    auto newType = llvm::FunctionType::get(
        builder.getVoidTy(), builder.getInt8PtrTy()->getPointerTo(),
        /*isVarArg=*/false);
    auto newName = makePackedFunctionName(func.getName());
    auto funcCst = module->getOrInsertFunction(newName, newType);
    llvm::Function *interfaceFunc =
        llvm::cast<llvm::Function>(funcCst.getCallee());
    interfaceFunctions.insert(interfaceFunc);

    // Extract the arguments from the type-erased argument list and cast them to
    // the proper types.
    auto bb = llvm::BasicBlock::Create(ctx);
    bb->insertInto(interfaceFunc);
    builder.SetInsertPoint(bb);
    llvm::Value *argList = interfaceFunc->arg_begin();
    llvm::SmallVector<llvm::Value *, 8> args;
    args.reserve(llvm::size(func.args()));
    for (auto &indexedArg : llvm::enumerate(func.args())) {
      llvm::Value *argIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), llvm::APInt(64, indexedArg.index()));
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
          builder.getInt64Ty(), llvm::APInt(64, llvm::size(func.args())));
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
    JIT_LOG("No object for " << m->getModuleIdentifier()
                             << " in cache. Compiling.\n");
    return nullptr;
  }
  JIT_LOG("Object for " << m->getModuleIdentifier() << " loaded from cache.\n");
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

JIT::JIT(Namespace &ns, bool enableObjectCache,
         bool enableGDBNotificationListener,
         bool enablePerfNotificationListener)
    : ns(ns), cache(enableObjectCache ? new ObjectCache() : nullptr),
      gdbListener(enableGDBNotificationListener
                      ? llvm::JITEventListener::createGDBRegistrationListener()
                      : nullptr),
      perfListener(enablePerfNotificationListener
                       ? llvm::JITEventListener::createPerfJITEventListener()
                       : nullptr){};

MaybeJIT JIT::make(Namespace &ns,
                   mlir::ArrayRef<llvm::StringRef> sharedLibPaths,
                   mlir::Optional<llvm::CodeGenOpt::Level> jitCodeGenOptLevel,
                   bool enableObjectCache, bool enableGDBNotificationListener,
                   bool enablePerfNotificationListener) {

  auto jitEngine = std::make_unique<JIT>(ns, enableObjectCache,
                                         enableGDBNotificationListener,
                                         enablePerfNotificationListener);

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

  auto maybeModule = jitEngine->ns.compileToLLVM();

  if (!maybeModule.hasValue()) {
    return llvm::None;
  }

  auto llvmModule = std::move(maybeModule.getValue());
  packFunctionArguments(llvmModule.get());

  auto dataLayout = llvmModule->getDataLayout();

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
    if (jitEngine->gdbListener)
      objectLayer->registerJITEventListener(*jitEngine->gdbListener);
    if (jitEngine->perfListener)
      objectLayer->registerJITEventListener(*jitEngine->perfListener);

    // COFF format binaries (Windows) need special handling to deal with
    // exported symbol visibility.
    // cf llvm/lib/ExecutionEngine/Orc/LLJIT.cpp LLJIT::createObjectLinkingLayer
    llvm::Triple targetTriple(llvm::Twine(llvmModule->getTargetTriple()));
    if (targetTriple.isOSBinFormatCOFF()) {
      objectLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      objectLayer->setAutoClaimResponsibilityForObjectSymbols(true);
    }

    // Resolve symbols from shared libraries.
    for (auto libPath : sharedLibPaths) {
      auto mb = llvm::MemoryBuffer::getFile(libPath);
      if (!mb) {
        llvm::errs() << "Failed to create MemoryBuffer for: " << libPath
                     << "\nError: " << mb.getError().message() << "\n";
        continue;
      }
      auto &JD    = session.createBareJITDylib(std::string(libPath));
      auto loaded = llvm::orc::DynamicLibrarySearchGenerator::Load(
          libPath.data(), dataLayout.getGlobalPrefix());
      if (!loaded) {
        llvm::errs() << "Could not load " << libPath << ":\n  "
                     << loaded.takeError() << "\n";
        continue;
      }

      JD.addGenerator(std::move(*loaded));
      cantFail(objectLayer->add(JD, std::move(mb.get())));
    }

    return objectLayer;
  };

  // Callback to inspect the cache and recompile on demand. This follows Lang's
  // LLJITWithObjectCache example.
  auto compileFunctionCreator = [&](llvm::orc::JITTargetMachineBuilder JTMB)
      -> llvm::Expected<
          std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
    if (jitCodeGenOptLevel)
      JTMB.setCodeGenOptLevel(jitCodeGenOptLevel.getValue());
    auto TM = JTMB.createTargetMachine();
    if (!TM)
      return TM.takeError();
    return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(
        std::move(*TM), jitEngine->cache.get());
  };

  // Create the LLJIT by calling the LLJITBuilder with 2 callbacks.
  auto jit =
      cantFail(llvm::orc::LLJITBuilder()
                   .setCompileFunctionCreator(compileFunctionCreator)
                   .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
                   .create());

  // Add a ThreadSafemodule to the engine and return.
  llvm::orc::ThreadSafeModule tsm(std::move(llvmModule), std::move(ctx));

  // TODO: Do we need a module transformer here???

  cantFail(jit->addIRModule(std::move(tsm)));
  jitEngine->engine = std::move(jit);

  // Resolve symbols that are statically linked in the current process.
  llvm::orc::JITDylib &mainJD = jitEngine->engine->getMainJITDylib();
  mainJD.addGenerator(
      cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          dataLayout.getGlobalPrefix())));

  return MaybeJIT(std::move(jitEngine));
};

llvm::Expected<void (*)(void **)> JIT::lookup(llvm::StringRef name) const {
  auto expectedSymbol = engine->lookup(makePackedFunctionName(name));

  // JIT lookup may return an Error referring to strings stored internally by
  // the JIT. If the Error outlives the ExecutionEngine, it would want have a
  // dangling reference, which is currently caught by an assertion inside JIT
  // thanks to hand-rolled reference counting. Rewrap the error message into a
  // string before returning. Alternatively, ORC JIT should consider copying
  // the string into the error message.
  if (!expectedSymbol) {
    std::string errorMessage;
    llvm::raw_string_ostream os(errorMessage);
    llvm::handleAllErrors(expectedSymbol.takeError(),
                          [&os](llvm::ErrorInfoBase &ei) { ei.log(os); });
    return make_string_error(os.str());
  }

  auto rawFPtr = expectedSymbol->getAddress();
  auto fptr    = reinterpret_cast<void (*)(void **)>(rawFPtr);
  if (!fptr)
    return make_string_error("looked up function is null");
  return fptr;
}

llvm::Error JIT::invokePacked(llvm::StringRef name,
                              llvm::MutableArrayRef<void *> args) {
  auto expectedFPtr = lookup(name);
  if (!expectedFPtr)
    return expectedFPtr.takeError();
  auto fptr = *expectedFPtr;

  (*fptr)(args.data());

  return llvm::Error::success();
}

void JIT::registerSymbols(
    llvm::function_ref<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)>
        symbolMap) {
  auto &mainJitDylib = engine->getMainJITDylib();
  cantFail(mainJitDylib.define(
      absoluteSymbols(symbolMap(llvm::orc::MangleAndInterner(
          mainJitDylib.getExecutionSession(), engine->getDataLayout())))));
}

} // namespace serene
