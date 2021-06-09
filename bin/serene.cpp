/**
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
 */

#include "serene/serene.h"
#include "serene/context.h"
#include "serene/namespace.h"
#include "serene/reader/reader.h"
#include "serene/reader/semantics.h"
#include "serene/slir/slir.h"
#include <iostream>
#include <llvm/Support/CommandLine.h>

using namespace std;
using namespace serene;

namespace cl = llvm::cl;

namespace {
enum Action { None, DumpAST, DumpIR, DumpSemantic };
}

static cl::opt<std::string> inputFile(cl::Positional,
                                      cl::desc("The Serene file to compile"),
                                      cl::init("-"),
                                      cl::value_desc("filename"));

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select what to dump."),
    cl::values(clEnumValN(DumpSemantic, "ast1",
                          "Output the AST after one level of analysis only")),
    cl::values(clEnumValN(DumpIR, "slir", "Output the SLIR only")),
    cl::values(clEnumValN(DumpAST, "ast", "Output the AST only"))

);

int main(int argc, char *argv[]) {
  cl::ParseCommandLineOptions(argc, argv, "Serene compiler \n");

  switch (emitAction) {
  case Action::DumpAST: {
    reader::FileReader *r = new reader::FileReader(inputFile);
    r->toString();
    delete r;
    return 0;
  };

  case Action::DumpSemantic: {
    reader::FileReader *r = new reader::FileReader(inputFile);

    auto maybeAst = r->read();

    if (!maybeAst) {
      throw std::move(maybeAst.getError());
    }
    auto &ast = maybeAst.getValue();

    auto ctx = makeSereneContext();
    auto ns = makeNamespace(*ctx, "user", llvm::None);
    auto afterAst = reader::analyze(*ctx, ast);

    if (afterAst) {
      dump(afterAst.getValue());
      delete r;
      return 0;
    } else {
      throw std::move(afterAst.getError());
    }

    delete r;
    return 0;
  };
  case Action::DumpIR: {
    reader::FileReader *r = new reader::FileReader(inputFile);

    auto maybeAst = r->read();

    if (!maybeAst) {
      throw std::move(maybeAst.getError());
    }

    // TODO: Move all this to a compile fn
    auto &ast = maybeAst.getValue();

    auto ctx = makeSereneContext();
    auto ns = makeNamespace(*ctx, "user", llvm::None);
    auto afterAst = reader::analyze(*ctx, ast);

    if (afterAst) {
      auto isSet = ns->setTree(afterAst.getValue());

      if (isSet.succeeded()) {
        ctx->insertNS(ns);
        serene::slir::dumpSLIR(*ctx, ns->name);
      } else {
        llvm::outs() << "Can't set the tree of the namespace!\n";
      }

      delete r;
      return 0;
    } else {
      throw std::move(afterAst.getError());
    }

    delete r;
    return 0;
  }
  default: {
    llvm::errs() << "No action specified. TODO: Print out help here";
  }
  }

  return 1;
}
