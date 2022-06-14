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

#include "serene/context.h"    // for SereneContext, makeSereneCon...
#include "serene/jit/halley.h" // for makeHalleyJIT, Engine, Maybe...

#include <llvm/ADT/StringRef.h>         // for StringRef
#include <llvm/Support/CommandLine.h>   // for list, cat, desc, MiscFlags
#include <llvm/Support/ManagedStatic.h> // for ManagedStatic
#include <llvm/Support/TargetSelect.h>  // for InitializeAllAsmParsers, Ini...

#include <string>  // for string
#include <utility> // for move

namespace serene {
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

  // #ifdef SERENE_WITH_MLIR_CL_OPTION
  //   // mlir::registerAsmPrinterCLOptions();
  //   mlir::registerMLIRContextCLOptions();
  //   mlir::registerPassManagerCLOptions();
  // #endif
}

void applySereneCLOptions(serene::jit::Engine &engine) {
  if (!options.isConstructed()) {
    return;
  }

  auto &ctx = engine.getContext();
  ctx.setLoadPaths(options->loadPaths);

  // #ifdef SERENE_WITH_MLIR_CL_OPTION
  //   mlir::applyPassManagerCLOptions(ctx.pm);
  // #endif
}

void initSerene() {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
};

serene::jit::MaybeEngine makeEngine(Options opts) {
  auto ctx = makeSereneContext(opts);
  return serene::jit::makeHalleyJIT(std::move(ctx));
};

} // namespace serene
