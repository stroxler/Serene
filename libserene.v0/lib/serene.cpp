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

#include "serene/serene.h"

#include "serene/diagnostics.h"
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

  auto loc = reader::LocationRange::UnknownLocation("nsname");

  // auto ns = ctx.importNamespace("docs.examples.hello_world", loc);

  // if (!ns) {
  //   auto es        = ns.getError();
  //   auto nsloadErr = errors::makeError(loc, errors::NSLoadError);
  //   es.push_back(nsloadErr);
  //   return exprs::MaybeNode::error(es);
  // }

  auto errs = ctx.jit->addAST(input);
  if (errs) {
    return errs;
  }

  //   auto e    = input[0];
  // auto *sym = llvm::dyn_cast<exprs::Symbol>(e.get());

  // if (sym == nullptr) {
  //   return exprs::makeErrorNode(e->location, errors::UnknownError, "only
  //   sym");
  // }

  // llvm::outs() << "Read: " << sym->toString() << "\n";

  // // Get the anonymous expression's JITSymbol.
  // auto symptr = ctx.jit->lookup(*sym);
  // if (!symptr) {
  //   return exprs::MaybeNode::error(symptr.getError());
  // }

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
