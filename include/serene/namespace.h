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

/// Serene's namespaces are the unit of compilation. Any code that needs to be
/// compiled has to be in a namespace. The official way to create a new
/// namespace is to use the `makeNamespace` function.
class Namespace : public WithTrait<Namespace, slir::Generatable> {
private:
  SereneContext &ctx;
  bool initialized             = false;
  std::atomic<uint> fn_counter = 0;
  exprs::Ast tree;
  std::unique_ptr<llvm::Module> llvmModule;
  mlir::ModuleOp module;

public:
  mlir::StringRef name;
  llvm::Optional<std::string> filename;

  /// The root environment of the namespace on the semantic analysis phase.
  /// Which is a mapping from names to AST nodes ( no evaluation ).
  Environment<std::string, exprs::Node> semanticEnv;

  Environment<llvm::StringRef, mlir::Value> symbolTable;
  Namespace(SereneContext &ctx, llvm::StringRef ns_name,
            llvm::Optional<llvm::StringRef> filename);

  exprs::Ast &getTree();
  mlir::LogicalResult setTree(exprs::Ast &);
  uint nextFnCounter();

  mlir::ModuleOp &getModule();
  SereneContext &getContext();
  void setLLVMModule(std::unique_ptr<llvm::Module>);
  llvm::Module &getLLVMModule();

  // Generatable Trait

  /// Generate the IR of the namespace with respect to the compilation phase
  mlir::LogicalResult generate();
  mlir::LogicalResult runPasses();

  /// Dumps the namespace with respect to the compilation phase
  void dump();

  ~Namespace();
};

std::shared_ptr<Namespace>
makeNamespace(SereneContext &ctx, llvm::StringRef name,
              llvm::Optional<llvm::StringRef> filename, bool setCurrent = true);
} // namespace serene

#endif
