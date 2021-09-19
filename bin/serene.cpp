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

#include "serene/config.h"
#include "serene/context.h"
#include "serene/jit.h"
#include "serene/namespace.h"
#include "serene/reader/location.h"
#include "serene/reader/reader.h"
#include "serene/reader/semantics.h"
#include "serene/slir/generatable.h"
#include "serene/slir/slir.h"

#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/Path.h>
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
  CompileToObject,
  Compile,
  // TODO: Remove this option and replace it by a subcommand
  RunJIT,
};
}
static std::string banner =
    llvm::formatv("\n\nSerene Compiler Version {0}"
                  "\nCopyright (C) 2019-2021 "
                  "Sameer Rahmani <lxsameer@gnu.org>\n"
                  "Serene comes with ABSOLUTELY NO WARRANTY;\n"
                  "This is free software, and you are welcome\n"
                  "to redistribute it under certain conditions; \n"
                  "for details take a look at the LICENSE file.\n",
                  SERENE_VERSION);

static cl::opt<std::string> inputNS(cl::Positional, cl::desc("<namespace>"),
                                    cl::Required);

static cl::opt<std::string> outputFile(
    "o", cl::desc("The relative path to the output file from the build dir"),
    cl::init("-"), cl::value_desc("filename"));

static cl::opt<std::string>
    outputDir("b", cl::desc("The absolute path to the build directory"),
              cl::value_desc("filename"), cl::Required);

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select what to dump."), cl::init(Compile),
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
                          "Compile to object file.")),
    cl::values(clEnumValN(Compile, "target",
                          "Compile to target code. (Default)")),
    cl::values(clEnumValN(RunJIT, "jit",
                          "Run the give input file with the JIT."))

);

llvm::cl::OptionCategory clOptionsCategory{"Discovery options"};
static cl::list<std::string>
    loadPaths("l", cl::desc("The load path to use for compilation."),
              llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::PositionalEatsArgs,
              llvm::cl::cat(clOptionsCategory));

int dumpAsObject(Namespace &ns) {
  // TODO: Move the compilation process to the Namespace class
  auto maybeModule = ns.compileToLLVM();
  // TODO: Fix this call to raise the wrapped error instead
  if (!maybeModule) {
    // TODO: Rais and error: "Faild to generato LLVM IR for namespace"
    return -1;
  }

  auto module = std::move(maybeModule.getValue());
  auto &ctx   = ns.getContext();

  // TODO: We need to set the triple data layout and everything to that sort in
  // one place. We want them for the JIT as well and also we're kinda
  // duplicating what we're doing in `Namespace#compileToLLVM`.
  module->setTargetTriple(ctx.targetTriple);

  std::string Error;
  auto target = llvm::TargetRegistry::lookupTarget(ctx.targetTriple, Error);

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
      target->createTargetMachine(ctx.targetTriple, cpu, features, opt, rm);
  auto targetMachine = std::unique_ptr<llvm::TargetMachine>(targetMachinePtr);

  module->setDataLayout(targetMachine->createDataLayout());

  auto filename =
      strcmp(outputFile.c_str(), "-") == 0 ? "output" : outputFile.c_str();

  std::error_code ec;
  llvm::SmallString<256> destFile(outputDir);
  llvm::sys::path::append(destFile, filename);
  auto destObjFilePath = llvm::formatv("{0}.o", destFile).str();
  llvm::raw_fd_ostream dest(destObjFilePath, ec, llvm::sys::fs::OF_None);

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

  pass.run(*module);
  dest.flush();

  if (emitAction == Action::Compile) {
    llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> opts =
        new clang::DiagnosticOptions;
    clang::DiagnosticsEngine diags(
        new clang::DiagnosticIDs, opts,
        new clang::TextDiagnosticPrinter(llvm::errs(), opts.get()));

    clang::driver::Driver d("clang", ctx.targetTriple, diags,
                            "Serene compiler");
    std::vector<const char *> args = {"serenec"};

    args.push_back(destObjFilePath.c_str());
    args.push_back("-o");
    args.push_back(destFile.c_str());

    d.setCheckInputsExist(false);

    std::unique_ptr<clang::driver::Compilation> compilation;
    compilation.reset(d.BuildCompilation(args));

    if (!compilation) {
      return 1;
    }

    llvm::SmallVector<std::pair<int, const clang::driver::Command *>>
        failCommand;
    // compilation->ExecuteJobs(compilation->getJobs(), failCommand);

    d.ExecuteCompilation(*compilation, failCommand);
    if (failCommand.empty()) {
      llvm::outs() << "Done!\n";
    } else {
      llvm::errs() << "Linking failed!\n";
    }
  }

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

  cl::ParseCommandLineOptions(argc, argv, banner);
  auto ctx    = makeSereneContext();
  auto userNS = makeNamespace(*ctx, "user", llvm::None);

  // TODO: We might want to find a better place for this
  applyPassManagerCLOptions(ctx->pm);
  ctx->sourceManager.setLoadPaths(loadPaths);

  auto runLoc = reader::LocationRange::UnknownLocation(inputNS);
  auto ns     = ctx->sourceManager.readNamespace(*ctx, inputNS, runLoc);

  if (!ns) {
    return (int)std::errc::no_such_file_or_directory;
  }

  // TODO: handle the outputDir by not forcing it. it should be
  //       default to the current working dir
  if (outputDir == "-") {
    llvm::errs() << "Error: The build directory is not set. Did you forget to "
                    "use '-build-dir'?\n";
    return 1;
  }

  switch (emitAction) {

  case Action::RunJIT: {
    // TODO: Replace it by a proper jit configuration
    ctx->setOperationPhase(CompilationPhase::NoOptimization);
    break;
  };

    // Just print out the raw AST
  case Action::DumpAST: {
    auto ast = ns->getTree();
    llvm::outs() << exprs::astToString(&ast) << "\n";
    return 0;
  };

  case Action::DumpSemantic: {
    auto ast      = ns->getTree();
    auto afterAst = reader::analyze(*ctx, ast);

    if (!afterAst) {
      throw std::move(afterAst.getError());
    }

    ast = afterAst.getValue();

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

  case Action::Compile: {
    ctx->setOperationPhase(CompilationPhase::NoOptimization);
    break;
  }

  default: {
    llvm::errs() << "No action specified. TODO: Print out help here\n";
    return 1;
  }
  }

  // Perform the semantic analytics
  auto afterAst = reader::analyze(*ctx, ns->getTree());
  if (!afterAst) {
    throw std::move(afterAst.getError());
  }

  auto isSet = ns->setTree(afterAst.getValue());
  if (isSet.succeeded()) {
    ctx->insertNS(ns);

    switch (emitAction) {
    case Action::DumpSLIR:
    case Action::DumpMLIR:
    case Action::DumpLIR: {
      ns->dump();
      break;
    };
    case Action::DumpIR: {
      auto maybeModule = ns->compileToLLVM();

      if (!maybeModule) {
        llvm::errs() << "Failed to generate the IR.\n";
        return 1;
      }

      maybeModule.getValue()->dump();
      break;
    };

    case Action::RunJIT: {
      auto maybeJIT = JIT::make(*ns.get());
      if (!maybeJIT) {
        // TODO: panic in here: "Couldn't creat the JIT!"
        return -1;
      }
      auto jit = std::move(maybeJIT.getValue());

      if (jit->invoke("main")) {
        llvm::errs() << "Faild to invoke the 'main' function.\n";
        return 1;
      }
      llvm::outs() << "Done!";
      break;
    };

    case Action::Compile:
    case Action::CompileToObject: {
      return dumpAsObject(*ns);
    };
    default: {
      llvm::errs() << "Action is not supported yet!\n";
    };
    }
  } else {
    llvm::errs() << "Can't set the tree of the namespace!\n";
  }

  return 0;
}
