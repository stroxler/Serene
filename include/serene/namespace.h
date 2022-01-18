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
 * - `environments` vector is the owner of all the semantic envs
 * - The first env in the `environments` is the root env.
 */

// TODO: Add a mechanism to figure out whether a namespace has changed or not
//       either on memory or disk

#ifndef SERENE_NAMESPACE_H
#define SERENE_NAMESPACE_H

#include "serene/environment.h"
#include "serene/errors/error.h"
#include "serene/export.h"
#include "serene/slir/generatable.h"
#include "serene/traits.h"
#include "serene/utils.h"

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Module.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>

#define NAMESPACE_LOG(...) \
  DEBUG_WITH_TYPE("NAMESPACE", llvm::dbgs() << __VA_ARGS__ << "\n");

namespace serene {
class SereneContext;
class Namespace;

namespace exprs {
class Expression;
using Node = std::shared_ptr<Expression>;
using Ast  = std::vector<Node>;
} // namespace exprs

using NSPtr                = std::shared_ptr<Namespace>;
using MaybeNS              = Result<NSPtr, errors::ErrorTree>;
using MaybeModule          = llvm::Optional<llvm::orc::ThreadSafeModule>;
using MaybeModuleOp        = llvm::Optional<mlir::OwningOpRef<mlir::ModuleOp>>;
using SemanticEnv          = Environment<std::string, exprs::Node>;
using SemanticEnvPtr       = std::unique_ptr<SemanticEnv>;
using SemanticEnvironments = std::vector<SemanticEnvPtr>;
using Form                 = std::pair<SemanticEnv, exprs::Ast>;
using Forms                = std::vector<Form>;

/// Serene's namespaces are the unit of compilation. Any code that needs to be
/// compiled has to be in a namespace. The official way to create a new
/// namespace is to use the `makeNamespace` member function of `SereneContext`.
class SERENE_EXPORT Namespace {
  friend SereneContext;

private:
  SereneContext &ctx;

  // Anonymous function counter. We need to assing a unique name to each
  // anonymous function and we use this counter to generate those names
  std::atomic<uint> fn_counter = 0;

  /// The content of the namespace. It should alway hold a semantically
  /// correct AST. It means thet the AST that we want to stor here has
  /// to pass the semantic analyzer.
  exprs::Ast tree;

  SemanticEnvironments environments;

  std::vector<llvm::StringRef> symbolList;

  /// Create a naw namespace with the given `name` and optional `filename` and
  /// return a shared pointer to it in the given Serene context.
  static NSPtr make(SereneContext &ctx, llvm::StringRef name,
                    llvm::Optional<llvm::StringRef> filename);

public:
  std::string name;
  llvm::Optional<std::string> filename;

  Namespace(SereneContext &ctx, llvm::StringRef ns_name,
            llvm::Optional<llvm::StringRef> filename);

  /// Create a new environment with the give \p parent as the parent,
  /// push the environment to the internal environment storage and
  /// return a reference to it. The namespace itself is the owner of
  /// environments.
  SemanticEnv &createEnv(SemanticEnv *parent);

  /// Return a referenece to the top level (root) environment of ns.
  SemanticEnv &getRootEnv();

  /// Define a new binding in the root environment with the given \p name
  /// and the given \p node. Defining a new binding with a name that
  /// already exists in legal and will overwrite the previous binding and
  /// the given name will point to a new value from now on.
  mlir::LogicalResult define(std::string &name, exprs::Node &node);

  /// Add the given \p ast to the namespace and return any possible error.
  /// The given \p ast will be added to a vector of ASTs that the namespace
  /// have. In a normal compilation a Namespace will have a vector of ASTs
  /// with only one element, but in a REPL like environment it might have
  /// many elements.
  ///
  /// This function runs the semantic analyzer on the \p ast as well.
  errors::OptionalErrors addTree(exprs::Ast &ast);
  exprs::Ast &getTree();

  const std::vector<llvm::StringRef> &getSymList() { return symbolList; };

  /// Increase the function counter by one
  uint nextFnCounter();

  SereneContext &getContext();

  /// Generate and return a MLIR ModuleOp tha contains the IR of the namespace
  /// with respect to the compilation phase
  MaybeModuleOp generate(unsigned offset = 0);

  /// Compile the namespace to a llvm module. It will call the
  /// `generate` method of the namespace to generate the IR.
  MaybeModule compileToLLVM();

  /// Compile the given namespace from the given \p offset of AST till the end
  /// of the trees.
  MaybeModule compileToLLVMFromOffset(unsigned offset);

  /// Run all the passes specified in the context on the given MLIR ModuleOp.
  mlir::LogicalResult runPasses(mlir::ModuleOp &m);

  /// Dumps the namespace with respect to the compilation phase
  void dump();

  void enqueueError(llvm::StringRef e) const;

  ~Namespace();
};

} // namespace serene

#endif
