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

#include "serene/context.h"
#include "serene/namespace.h"
#include "serene/reader/reader.h"
#include "serene/reader/semantics.h"
#include "serene/slir/generatable.h"
#include "serene/slir/slir.h"

#include <iostream>
#include <llvm/Support/CommandLine.h>
#include <memory>

using namespace std;
using namespace serene;

namespace cl = llvm::cl;

namespace {
enum Action { None, DumpAST, DumpIR, DumpSLIR, DumpMLIR, DumpSemantic };
}

static cl::opt<std::string> inputFile(cl::Positional,
                                      cl::desc("The Serene file to compile"),
                                      cl::init("-"),
                                      cl::value_desc("filename"));

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select what to dump."),
    cl::values(clEnumValN(DumpSemantic, "semantic",
                          "Output the AST after one level of analysis only")),
    cl::values(clEnumValN(DumpIR, "ir", "Output the lowered IR only")),
    cl::values(clEnumValN(DumpSLIR, "slir", "Output the SLIR only")),
    cl::values(clEnumValN(DumpMLIR, "mlir",
                          "Output the MLIR only (Lowered SLIR)")),
    cl::values(clEnumValN(DumpAST, "ast", "Output the AST only"))

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

int main(int argc, char *argv[]) {
  // mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

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

  default: {
    llvm::errs() << "No action specified. TODO: Print out help here\n";
    return 1;
  }
  }

  auto afterAst = readAndAnalyze(*ctx);
  auto isSet    = ns->setTree(afterAst);

  if (isSet.succeeded()) {
    ctx->insertNS(ns);
    serene::slir::dump<Namespace>(*ns);
  } else {
    llvm::outs() << "Can't set the tree of the namespace!\n";
  }

  return 0;
}
