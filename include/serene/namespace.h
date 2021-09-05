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

#ifndef SERENE_NAMESPACE_H
#define SERENE_NAMESPACE_H

#include "serene/environment.h"
#include "serene/slir/generatable.h"
#include "serene/traits.h"
#include "serene/utils.h"

#include <algorithm>
#include <atomic>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Module.h>
#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
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
  // TODO: replace the temporary `bool` by errors::Error
using MaybeModule = Result<std::unique_ptr<llvm::Module>, bool>;

// TODO: replace the temporary `bool` by errors::Error
using MaybeModuleOp = Result<mlir::OwningOpRef<mlir::ModuleOp>, bool>;

/// Serene's namespaces are the unit of compilation. Any code that needs to be
/// compiled has to be in a namespace. The official way to create a new
/// namespace is to use the `makeNamespace` function.
class Namespace {
private:
  SereneContext &ctx;
  bool initialized = false;

  // Anonymous function counter. We need to assing a unique name to each
  // anonymous function and we use this counter to generate those names
  std::atomic<uint> fn_counter = 0;

  // The content of the namespace
  exprs::Ast tree;

public:
  mlir::StringRef name;
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
  mlir::LogicalResult setTree(exprs::Ast &);

  /// Increase the function counter by one
  uint nextFnCounter();

  SereneContext &getContext();

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

  ~Namespace();
};

using NSPtr = std::shared_ptr<Namespace>;
/// Create a naw namespace with the given `name` and optional `filename` and
/// return a shared pointer to it in the given Serene context. If the
/// `setCurrent` argument is set to true, the created NS will become the curret
/// namespace in the context
NSPtr makeNamespace(SereneContext &ctx, llvm::StringRef name,
                    llvm::Optional<llvm::StringRef> filename,
                    bool setCurrent = true);

} // namespace serene

#endif
