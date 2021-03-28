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

#include "serene/serene.hpp"
#include "serene/reader.hpp"
#include "serene/sir/sir.hpp"
#include <iostream>
#include <llvm/Support/CommandLine.h>

using namespace std;
using namespace serene;

namespace cl = llvm::cl;

namespace {
enum Action { None, DumpAST, DumpIR };
}

static cl::opt<std::string> inputFile(cl::Positional,
                                      cl::desc("The Serene file to compile"),
                                      cl::init("-"),
                                      cl::value_desc("filename"));

static cl::opt<enum Action>
    emitAction("emit", cl::desc("Select what to dump."),
               cl::values(clEnumValN(DumpIR, "sir", "Output the SLIR only")),
               cl::values(clEnumValN(DumpAST, "ast", "Output the AST only")));

int main(int argc, char *argv[]) {
  cl::ParseCommandLineOptions(argc, argv, "Serene compiler \n");

  switch (emitAction) {
  case Action::DumpAST: {
    FileReader *r = new FileReader(inputFile);
    r->dumpAST();
    delete r;
    return 0;
  }
  case Action::DumpIR: {
    FileReader *r = new FileReader(inputFile);

    serene::sir::dumpSIR(*r->read());
    delete r;
    return 0;
  }
  default: {
    llvm::errs() << "No action specified. TODO: Print out help here";
  }
  }

  return 1;
}
