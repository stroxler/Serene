/* -*- C++ -*-
 * Serene Programming Language
 *
 * Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
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

/**
 * Commentary:
 * Rules of a namespace:
 * - A namespace has have a name and it has to own it.
 * - A namespace may or may not be assiciated with a file
 * - The internal AST of a namespace is an evergrowing tree which may expand at
 *   any given time. For example via iteration of a REPL
 */

#ifndef SERENE_NAMESPACE_H
#define SERENE_NAMESPACE_H

#include "serene/environment.h"
#include "serene/errors/error.h"
#include "serene/export.h"
#include "serene/slir/generatable.h"
#include "serene/traits.h"
#include "serene/utils.h"

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Module.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <atomic>
#include <memory>
#include <string>

#define NAMESPACE_LOG(...) \
  DEBUG_WITH_TYPE("NAMESPACE", llvm::dbgs() << __VA_ARGS__ << "\n");

namespace serene {
class SereneContext;

namespace exprs {
class Expression;
using Node = std::shared_ptr<Expression>;
using Ast  = std::vector<Node>;
} // namespace exprs

using MaybeModule   = llvm::Optional<llvm::orc::ThreadSafeModule>;
using MaybeModuleOp = llvm::Optional<mlir::OwningOpRef<mlir::ModuleOp>>;

/// Serene's namespaces are the unit of compilation. Any code that needs to be
/// compiled has to be in a namespace. The official way to create a new
/// namespace is to use the `makeNamespace` function.
class SERENE_EXPORT Namespace {
private:
  SereneContext &ctx;

  // Anonymous function counter. We need to assing a unique name to each
  // anonymous function and we use this counter to generate those names
  std::atomic<uint> fn_counter = 0;

  /// The content of the namespace. It should alway hold a semantically
  /// correct AST. It means thet the AST that we want to stor here has
  /// to pass the semantic analyzer.
  exprs::Ast tree;

public:
  std::string name;
  llvm::Optional<std::string> filename;

  /// The root environment of the namespace on the semantic analysis phase.
  /// Which is a mapping from names to AST nodes ( no evaluation ).
  Environment<std::string, exprs::Node> semanticEnv;

  /// Th root environmanet to store the MLIR value during the IR generation
  /// phase.
  Environment<llvm::StringRef, mlir::Value> symbolTable;

  Namespace(SereneContext &ctx, llvm::StringRef ns_name,
            llvm::Optional<llvm::StringRef> filename);

  exprs::Ast &getTree();

  /// Expand the current tree of the namespace with the given \p ast by
  /// semantically analazing it first. If the give \p ast in not valid
  /// it will return analysis errors.
  errors::OptionalErrors expandTree(exprs::Ast &ast);

  /// Increase the function counter by one
  uint nextFnCounter();

  SereneContext &getContext();

  // TODO: Fix the return type and use a `llvm::Optional` instead
  /// Generate and return a MLIR ModuleOp tha contains the IR of the namespace
  /// with respect to the compilation phase
  MaybeModuleOp generate();

  /// Compile the namespace to a llvm module. It will call the
  /// `generate` method of the namespace to generate the IR.
  MaybeModule compileToLLVM();

  /// Run all the passes specified in the context on the given MLIR ModuleOp.
  mlir::LogicalResult runPasses(mlir::ModuleOp &m);

  /// Dumps the namespace with respect to the compilation phase
  void dump();

  void enqueueError(llvm::StringRef e) const;

  ~Namespace();
};

using NSPtr = std::shared_ptr<Namespace>;

using MaybeNS = Result<NSPtr, errors::ErrorTree>;
/// Create a naw namespace with the given `name` and optional `filename` and
/// return a shared pointer to it in the given Serene context. If the
/// `setCurrent` argument is set to true, the created NS will become the
/// curret namespace in the context
SERENE_EXPORT NSPtr makeNamespace(SereneContext &ctx, llvm::StringRef name,
                                  llvm::Optional<llvm::StringRef> filename,
                                  bool setCurrent = true);

} // namespace serene

#endif
