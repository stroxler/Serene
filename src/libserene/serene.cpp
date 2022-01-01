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
 */

#include "serene/serene.h"

#include "serene/diagnostics.h"
#include "serene/errors/constants.h"
#include "serene/exprs/expression.h"

// TODO: Remove it
#include "serene/exprs/number.h"
#include "serene/jit/halley.h"
#include "serene/reader/reader.h"
#include "serene/utils.h"

#include <llvm/ADT/None.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

namespace serene {
using exprs::Number;

void initCompiler() {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
};

// CLI Option ----------------

/// All the global CLI option ar defined here. If you need to add a new global
/// option
///  make sure that you are handling it in `applySereneCLOptions` too.
struct SereneOptions {

  llvm::cl::OptionCategory clOptionsCategory{"Discovery options"};

  llvm::cl::list<std::string> loadPaths{
      "l", llvm::cl::desc("The load path to use for compilation."),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::PositionalEatsArgs,
      llvm::cl::cat(clOptionsCategory)};

  llvm::cl::list<std::string> sharedLibraryPaths{
      "sl", llvm::cl::desc("Where to find shared libraries"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::PositionalEatsArgs,
      llvm::cl::cat(clOptionsCategory)};
};

static llvm::ManagedStatic<SereneOptions> options;

void registerSereneCLOptions() {
  // Make sure that the options struct has been constructed.
  *options;

#ifdef SERENE_WITH_MLIR_CL_OPTION
  // mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
#endif
}

void applySereneCLOptions(SereneContext &ctx) {
  if (!options.isConstructed()) {
    return;
  }

  ctx.sourceManager.setLoadPaths(options->loadPaths);

#ifdef SERENE_WITH_MLIR_CL_OPTION
  mlir::applyPassManagerCLOptions(ctx.pm);
#endif
}

SERENE_EXPORT exprs::MaybeAst read(SereneContext &ctx, std::string &input) {
  auto &currentNS = ctx.getCurrentNS();
  auto filename =
      !currentNS.filename.hasValue()
          ? llvm::None
          : llvm::Optional<llvm::StringRef>(currentNS.filename.getValue());

  return reader::read(ctx, input, currentNS.name, filename);
};

SERENE_EXPORT exprs::MaybeNode eval(SereneContext &ctx, exprs::Ast &input) {

  // TODO: Fix the eval function
  UNUSED(input);

  auto loc = reader::LocationRange::UnknownLocation("nsname");

  auto ns = ctx.importNamespace("docs.examples.hello_world", loc);

  if (!ns) {
    auto es        = ns.getError();
    auto nsloadErr = errors::makeError(loc, errors::NSLoadError);
    es.push_back(nsloadErr);
    return exprs::MaybeNode::error(es);
  }

  auto e    = input[0];
  auto *sym = llvm::dyn_cast<exprs::Symbol>(e.get());

  if (sym == nullptr) {
    return exprs::makeErrorNode(e->location, errors::UnknownError, "only sym");
  }

  llvm::outs() << "Read: " << sym->toString() << "\n";

  // Get the anonymous expression's JITSymbol.
  auto symptr = ctx.jit->lookup(*sym);
  if (!symptr) {
    return exprs::MaybeNode::error(symptr.getError());
  }

  llvm::outs() << "eval here\n";

  // sym((void **)3);

  // err = ctx.jit->addAst(input);
  // if (err) {
  //   llvm::errs() << err;
  //   auto e = errors::makeErrorTree(loc, errors::NSLoadError);

  //   return exprs::makeErrorNode(loc, errors::NSLoadError);
  // }
  return exprs::make<exprs::Number>(loc, "4", false, false);
};

SERENE_EXPORT void print(SereneContext &ctx, const exprs::Ast &input,
                         std::string &result) {
  UNUSED(ctx);
  result = exprs::astToString(&input);
};

} // namespace serene
