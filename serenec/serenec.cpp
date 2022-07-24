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

#include "serene/config.h"
#include "serene/context.h"
#include "serene/serene.h"
#include "serene/types/types.h"

// #include "serene/jit/halley.h"
// #include "serene/namespace.h"
// #include "serene/reader/location.h"
// #include "serene/reader/reader.h"
// #include "serene/semantics.h"
// #include "serene/serene.h"
// #include "serene/slir/generatable.h"
// #include "serene/slir/slir.h"

// #include <lld/Common/Driver.h>

// #include <clang/Driver/Compilation.h>
// #include <clang/Driver/Driver.h>
// #include <clang/Frontend/TextDiagnosticPrinter.h>
// #include <clang/Tooling/Tooling.h>
// #include <llvm/ADT/ArrayRef.h>
// #include <llvm/ADT/SmallString.h>
// #include <llvm/ADT/StringRef.h>
// #include <llvm/IR/LegacyPassManager.h>
// //#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/CommandLine.h>
// #include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormatVariadic.h>

#include <iostream>
// #include <llvm/Support/Host.h>
// #include <llvm/Support/Path.h>
// #include <llvm/Support/raw_ostream.h>
// #include <llvm/Target/TargetMachine.h>
// #include <llvm/Target/TargetOptions.h>

// #include <memory>

// using namespace std;
using namespace serene;

namespace cl = llvm::cl;

// namespace {
// enum Action {
//   None,
//   DumpAST,
//   DumpSLIR,
//   DumpMLIR,
//   DumpSemantic,
//   DumpLIR,
//   DumpIR,
//   CompileToObject,
//   Compile,
//   // TODO: Remove this option and replace it by a subcommand
//   RunJIT,
// };
// } // namespace

static std::string banner =
    llvm::formatv("\n\nSerene Compiler Version {0}"
                  "\nCopyright (C) 2019-2022 "
                  "Sameer Rahmani <lxsameer@gnu.org>\n"
                  "Serene comes with ABSOLUTELY NO WARRANTY;\n"
                  "This is free software, and you are welcome\n"
                  "to redistribute it under certain conditions; \n"
                  "for details take a look at the LICENSE file.\n",
                  SERENE_VERSION);

// static cl::opt<std::string> inputNS(cl::Positional, cl::desc("<namespace>"),
//                                     cl::Required);

// static cl::opt<std::string> outputFile(
//     "o", cl::desc("The relative path to the output file from the build dir"),
//     cl::init("-"), cl::value_desc("filename"));

// static cl::opt<std::string>
//     outputDir("b", cl::desc("The absolute path to the build directory"),
//               cl::value_desc("dir"), cl::Required);

// static cl::opt<enum Action> emitAction(
//     "emit", cl::desc("Select what to dump."), cl::init(Compile),
//     cl::values(clEnumValN(DumpSemantic, "semantic",
//                           "Output the AST after one level of analysis
//                           only")),
//     cl::values(clEnumValN(DumpIR, "ir", "Output the lowered IR only")),
//     cl::values(clEnumValN(DumpSLIR, "slir", "Output the SLIR only")),
//     cl::values(clEnumValN(DumpMLIR, "mlir",
//                           "Output the MLIR only (Lowered SLIR)")),
//     cl::values(clEnumValN(DumpLIR, "lir",
//                           "Output the LIR only (Lowerd to LLVM dialect)")),
//     cl::values(clEnumValN(DumpAST, "ast", "Output the AST only")),
//     cl::values(clEnumValN(CompileToObject, "object",
//                           "Compile to object file.")),
//     cl::values(clEnumValN(Compile, "target",
//                           "Compile to target code. (Default)")),
//     cl::values(clEnumValN(RunJIT, "jit",
//                           "Run the give input file with the JIT."))

// );

// int dumpAsObject(Namespace &ns) {
//   // TODO: Move the compilation process to the Namespace class
//   auto maybeModule = ns.compileToLLVM();
//   // TODO: Fix this call to raise the wrapped error instead
//   if (!maybeModule) {
//     // TODO: Rais and error: "Faild to generato LLVM IR for namespace"
//     return -1;
//   }

//   auto module = std::move(maybeModule.getValue());
//   auto &ctx   = ns.getContext();

//   // TODO: We need to set the triple data layout and everything to that sort
//   in
//   // one place. We want them for the JIT as well and also we're kinda
//   // duplicating what we're doing in `Namespace#compileToLLVM`.
//   module->setTargetTriple(ctx.targetTriple);

//   std::string Error;
//   const auto *target =
//       llvm::TargetRegistry::lookupTarget(ctx.targetTriple, Error);

//   // Print an error and exit if we couldn't find the requested target.
//   // This generally occurs if we've forgotten to initialise the
//   // TargetRegistry or we have a bogus target triple.
//   if (target == nullptr) {
//     llvm::errs() << Error;
//     return 1;
//   }

//   const auto *cpu      = "generic";
//   const auto *features = "";

//   llvm::TargetOptions opt;
//   auto rm = llvm::Optional<llvm::Reloc::Model>();
//   auto *targetMachinePtr =
//       target->createTargetMachine(ctx.targetTriple, cpu, features, opt, rm);
//   auto targetMachine =
//   std::unique_ptr<llvm::TargetMachine>(targetMachinePtr);

//   module->setDataLayout(targetMachine->createDataLayout());

//   const auto *filename =
//       strcmp(outputFile.c_str(), "-") == 0 ? "output" : outputFile.c_str();

//   std::error_code ec;
//   const auto pathSize(256);

//   llvm::SmallString<pathSize> destFile(outputDir);
//   llvm::sys::path::append(destFile, filename);
//   auto destObjFilePath = llvm::formatv("{0}.o", destFile).str();
//   llvm::raw_fd_ostream dest(destObjFilePath, ec, llvm::sys::fs::OF_None);

//   if (ec) {
//     llvm::errs() << "Could not open file: " << destObjFilePath;
//     llvm::errs() << "Could not open file: " << ec.message();
//     return 1;
//   }

//   llvm::legacy::PassManager pass;
//   auto fileType = llvm::CGFT_ObjectFile;

//   if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, fileType)) {
//     llvm::errs() << "TheTargetMachine can't emit a file of this type";
//     return 1;
//   }

//   pass.run(*module);
//   dest.flush();

//   if (emitAction == Action::Compile) {
//     std::vector<const char *> args = {"serenec"};

//     args.push_back("--eh-frame-hdr");
//     args.push_back("-m");
//     args.push_back("elf_x86_64");
//     args.push_back("-dynamic-linker");
//     args.push_back("/lib64/ld-linux-x86-64.so.2");
//     args.push_back(
//         "/usr/lib/gcc/x86_64-pc-linux-gnu/11.2.0/../../../../lib64/crt1.o");
//     args.push_back(
//         "/usr/lib/gcc/x86_64-pc-linux-gnu/11.2.0/../../../../lib64/crti.o");
//     args.push_back("/usr/lib/gcc/x86_64-pc-linux-gnu/11.2.0/crtbegin.o");
//     args.push_back("-L");
//     args.push_back("/usr/lib/gcc/x86_64-pc-linux-gnu/11.2.0/");
//     args.push_back("-L");
//     args.push_back("/usr/lib64/");

//     args.push_back(destObjFilePath.c_str());
//     args.push_back("-o");
//     args.push_back(destFile.c_str());
//     args.push_back("-lgcc");
//     args.push_back("--as-needed");
//     args.push_back("-lgcc_s");
//     args.push_back("--no-as-needed");
//     args.push_back("-lc");
//     args.push_back("-lgcc");
//     args.push_back("--as-needed");
//     args.push_back("-lgcc_s");
//     args.push_back("--no-as-needed");
//     args.push_back("/usr/lib/gcc/x86_64-pc-linux-gnu/11.2.0/crtend.o");
//     args.push_back(
//         "/usr/lib/gcc/x86_64-pc-linux-gnu/11.2.0/../../../../lib64/crtn.o");

//     lld::elf::link(args, false, llvm::outs(), llvm::errs());

//     //   llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> opts =
//     //       new clang::DiagnosticOptions;
//     //   clang::DiagnosticsEngine diags(
//     //       new clang::DiagnosticIDs, opts,
//     //       new clang::TextDiagnosticPrinter(llvm::errs(), opts.get()));

//     //   clang::driver::Driver d("clang", ctx.targetTriple, diags,
//     //                           "Serene compiler");
//     //   std::vector<const char *> args = {"serenec"};

//     //   args.push_back(destObjFilePath.c_str());
//     //   args.push_back("-o");
//     //   args.push_back(destFile.c_str());

//     //   d.setCheckInputsExist(true);

//     //   std::unique_ptr<clang::driver::Compilation> compilation;
//     //   compilation.reset(d.BuildCompilation(args));

//     //   if (!compilation) {
//     //     llvm::errs() << "can't create the compilation!\n";
//     //     return 1;
//     //   }

//     //   llvm::SmallVector<std::pair<int, const clang::driver::Command *>>
//     //       failCommand;

//     //   d.ExecuteCompilation(*compilation, failCommand);

//     //   if (failCommand.empty()) {
//     //     llvm::outs() << "Done!\n";
//     //   } else {
//     //     llvm::errs() << "Linking failed!\n";
//     //     failCommand.front().second->Print(llvm::errs(), "\n", false);
//     //   }
//     // }

//     return 0;
//   };

int main(int argc, char *argv[]) {
  SERENE_INIT();

  registerSereneCLOptions();

  cl::ParseCommandLineOptions(argc, argv, banner);

  auto maybeEngine = makeEngine();

  if (!maybeEngine) {
    llvm::errs() << "Error: Couldn't create the engine due to '"
                 << maybeEngine.takeError() << "'\n";
    return 1;
  }

  auto &engine = *maybeEngine;
  applySereneCLOptions(*engine);

  const std::string forms{"some.ns/sym"};
  const types::InternalString data(forms.c_str(), forms.size());

  auto err = engine->createEmptyNS("some.ns");

  if (err) {
    llvm::errs() << "Error: " << err << "'\n";
    return 1;
  }

  // err = engine->loadModule("some.ns", "/home/lxsameer/test.ll");

  // if (err) {
  //   llvm::errs() << "Error: " << err << "'\n";
  //   return 1;
  // }

  std::string core = "serene.core";
  auto maybeCore   = engine->loadNamespace(core);
  if (!maybeCore) {
    llvm::errs() << "Error: " << maybeCore.takeError() << "'\n";
    return 1;
  }

  auto bt = engine->lookup("serene.core", "compile");

  if (!bt) {
    llvm::errs() << "Error: " << bt.takeError() << "'\n";
    return 1;
  }

  if (*bt == nullptr) {
    llvm::errs() << "Error: nullptr?\n";
    return 1;
  }
  auto *c = *bt;

  (void)c;
  // void *res = c();
  // // for (int i = 0; i <= 10; i++) {
  // //   printf(">> %02x", *(c + i));
  // // }
  // printf("Res >> %p\n", res);
  // llvm::outs() << "Res: " << *((int *)res) << "\n";
  // (void)res;

  // // TODO: handle the outputDir by not forcing it. it should be
  // //       default to the current working dir
  // if (outputDir == "-") {
  //   llvm::errs() << "Error: The build directory is not set. Did you
  //   forget to
  //   "
  //                   "use '-b'?\n";
  //   return 1;
  // }

  // switch (emitAction) {

  // case Action::RunJIT: {
  //   // TODO: Replace it by a proper jit configuration
  //   ctx->setOperationPhase(CompilationPhase::NoOptimization);
  //   break;
  // };

  //   // Just print out the raw AST
  // case Action::DumpAST: {
  //   ctx->setOperationPhase(CompilationPhase::Parse);
  //   break;
  // };

  // case Action::DumpSemantic: {
  //   ctx->setOperationPhase(CompilationPhase::Analysis);
  //   break;
  // };

  // case Action::DumpSLIR: {
  //   ctx->setOperationPhase(CompilationPhase::SLIR);
  //   break;
  // }

  // case Action::DumpMLIR: {
  //   ctx->setOperationPhase(CompilationPhase::MLIR);
  //   break;
  // }

  // case Action::DumpLIR: {
  //   ctx->setOperationPhase(CompilationPhase::LIR);
  //   break;
  // }

  // case Action::DumpIR: {
  //   ctx->setOperationPhase(CompilationPhase::IR);
  //   break;
  // }

  // case Action::CompileToObject: {
  //   ctx->setOperationPhase(CompilationPhase::NoOptimization);
  //   break;
  // }

  // case Action::Compile: {
  //   ctx->setOperationPhase(CompilationPhase::NoOptimization);
  //   break;
  // }

  // default: {
  //   llvm::errs() << "No action specified. TODO: Print out help here\n";
  //   return 1;
  // }
  // }

  // auto runLoc  = reader::LocationRange::UnknownLocation(inputNS);
  // auto maybeNS = ctx->importNamespace(inputNS, runLoc);

  // if (!maybeNS) {
  //   auto err = maybeNS.takeError();
  //   throwErrors(*ctx, err);
  //   return (int)std::errc::no_such_file_or_directory;
  // }

  // auto ns = *maybeNS;

  // switch (emitAction) {
  // case Action::DumpAST:
  // case Action::DumpSemantic: {
  //   auto ast = ns->getTree();
  //   llvm::outs() << exprs::astToString(&ast) << "\n";
  //   return 0;
  // }

  // case Action::DumpSLIR:
  // case Action::DumpMLIR:
  // case Action::DumpLIR: {
  //   ns->dump();
  //   break;
  // };
  // case Action::DumpIR: {
  //   auto maybeModule = ns->compileToLLVM();

  //   if (!maybeModule) {
  //     llvm::errs() << "Failed to generate the IR.\n";
  //     return 1;
  //   }

  //   auto tsm = std::move(*maybeModule);
  //   tsm.withModuleDo([](auto &m) { m.dump(); });

  //   break;
  // };

  // // case Action::RunJIT: {
  // //   auto maybeJIT = JIT::make(*ns);
  // //   if (!maybeJIT) {
  // //     // TODO: panic in here: "Couldn't creat the JIT!"
  // //     return -1;
  // //   }
  // //   auto jit = std::move(maybeJIT.getValue());

  // //   if (jit->invoke("main")) {
  // //     llvm::errs() << "Faild to invoke the 'main' function.\n";
  // //     return 1;
  // //   }
  // //   llvm::outs() << "Done!";
  // //   break;
  // // };

  // // case Action::Compile:
  // // case Action::CompileToObject: {
  // //   return dumpAsObject(*ns);
  // // };
  // default: {
  //   llvm::errs() << "Action is not supported yet!\n";
  // };
  // }

  return 0;
}
