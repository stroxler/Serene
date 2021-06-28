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

#include "serene/serene.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "serene/context.h"
#include "serene/namespace.h"
#include "serene/reader/reader.h"
#include "serene/reader/semantics.h"
#include "serene/slir/generatable.h"
#include "serene/slir/slir.h"

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <memory>

using namespace std;
using namespace serene;

namespace cl = llvm::cl;

namespace {
enum Action {
  None,
  DumpAST,
  DumpSLIR,
  DumpMLIR,
  DumpSemantic,
  DumpLIR,
  DumpIR,
  CompileToObject
};
}

static cl::opt<std::string> inputFile(cl::Positional,
                                      cl::desc("The Serene file to compile"),
                                      cl::init("-"),
                                      cl::value_desc("filename"));

static cl::opt<std::string> outputFile("o",
                                       cl::desc("The path to the output file"),
                                       cl::init("-"),
                                       cl::value_desc("filename"));

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select what to dump."), cl::init(CompileToObject),
    cl::values(clEnumValN(DumpSemantic, "semantic",
                          "Output the AST after one level of analysis only")),
    cl::values(clEnumValN(DumpIR, "ir", "Output the lowered IR only")),
    cl::values(clEnumValN(DumpSLIR, "slir", "Output the SLIR only")),
    cl::values(clEnumValN(DumpMLIR, "mlir",
                          "Output the MLIR only (Lowered SLIR)")),
    cl::values(clEnumValN(DumpLIR, "lir",
                          "Output the LIR only (Lowerd to LLVM dialect)")),
    cl::values(clEnumValN(DumpAST, "ast", "Output the AST only")),
    cl::values(clEnumValN(CompileToObject, "object",
                          "Compile to object file. (Default)"))

);

exprs::Ast readInputFile() {
  auto r        = make_unique<reader::FileReader>(inputFile);
  auto maybeAst = r->read();

  if (!maybeAst) {
    throw std::move(maybeAst.getError());
  }
  return maybeAst.getValue();
};

exprs::Ast readAndAnalyze(SereneContext &ctx) {
  auto ast      = readInputFile();
  auto afterAst = reader::analyze(ctx, ast);

  if (!afterAst) {
    throw std::move(afterAst.getError());
  }

  return afterAst.getValue();
};

int dumpAsObject(Namespace &ns) {

  auto &module      = ns.getLLVMModule();
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  module.setTargetTriple(targetTriple);

  std::string Error;
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple, Error);

  // Print an error and exit if we couldn't find the requested target.
  // This generally occurs if we've forgotten to initialise the
  // TargetRegistry or we have a bogus target triple.
  if (!target) {
    llvm::errs() << Error;
    return 1;
  }

  auto cpu      = "generic";
  auto features = "";

  llvm::TargetOptions opt;
  auto rm = llvm::Optional<llvm::Reloc::Model>();
  auto targetMachinePtr =
      target->createTargetMachine(targetTriple, cpu, features, opt, rm);
  auto targetMachine = std::unique_ptr<llvm::TargetMachine>(targetMachinePtr);

  module.setDataLayout(targetMachine->createDataLayout());

  auto filename =
      strcmp(outputFile.c_str(), "-") == 0 ? "output.o" : outputFile.c_str();

  std::error_code ec;
  llvm::raw_fd_ostream dest(filename, ec, llvm::sys::fs::OF_None);

  if (ec) {
    llvm::errs() << "Could not open file: " << ec.message();
    return 1;
  }

  llvm::legacy::PassManager pass;
  auto fileType = llvm::CGFT_ObjectFile;

  if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, fileType)) {
    llvm::errs() << "TheTargetMachine can't emit a file of this type";
    return 1;
  }

  pass.run(module);
  dest.flush();

  llvm::outs() << "Wrote " << filename << "\n";

  return 0;
};

int main(int argc, char *argv[]) {
  // mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  cl::ParseCommandLineOptions(argc, argv, "Serene compiler \n");
  auto ctx = makeSereneContext();
  auto ns  = makeNamespace(*ctx, "user", llvm::None);
  // TODO: We might want to find a better place for this
  applyPassManagerCLOptions(ctx->pm);
  switch (emitAction) {

    // Just print out the raw AST
  case Action::DumpAST: {
    auto ast = readInputFile();
    llvm::outs() << exprs::astToString(&ast) << "\n";
    return 0;
  };

  case Action::DumpSemantic: {
    auto ast = readAndAnalyze(*ctx);
    llvm::outs() << exprs::astToString(&ast) << "\n";
    return 0;
  };

  case Action::DumpSLIR: {
    ctx->setOperationPhase(CompilationPhase::SLIR);
    break;
  }

  case Action::DumpMLIR: {
    ctx->setOperationPhase(CompilationPhase::MLIR);
    break;
  }

  case Action::DumpLIR: {
    ctx->setOperationPhase(CompilationPhase::LIR);
    break;
  }

  case Action::DumpIR: {
    ctx->setOperationPhase(CompilationPhase::IR);
    break;
  }

  case Action::CompileToObject: {
    ctx->setOperationPhase(CompilationPhase::NoOptimization);
    break;
  }

  default: {
    llvm::errs() << "No action specified. TODO: Print out help here\n";
    return 1;
  }
  }

  auto afterAst = readAndAnalyze(*ctx);
  auto isSet    = ns->setTree(afterAst);

  if (isSet.succeeded()) {
    ctx->insertNS(ns);
    if (mlir::failed(serene::slir::generate<Namespace>(*ns))) {
      llvm::errs() << "IR generation faild\n";
      return 1;
    }
    if (emitAction < CompileToObject) {
      serene::slir::dump<Namespace>(*ns);
    } else {
      return dumpAsObject(*ns);
    }
  } else {
    llvm::outs() << "Can't set the tree of the namespace!\n";
  }

  return 0;
}
