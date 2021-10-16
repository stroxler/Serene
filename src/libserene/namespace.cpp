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

#include "serene/namespace.h"

#include "serene/context.h"
#include "serene/errors/constants.h"
#include "serene/exprs/expression.h"
#include "serene/llvm/IR/Value.h"
#include "serene/reader/semantics.h"
#include "serene/slir/slir.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Support/LogicalResult.h>
#include <stdexcept>
#include <string>

using namespace std;
using namespace llvm;

namespace serene {

Namespace::Namespace(SereneContext &ctx, llvm::StringRef ns_name,
                     llvm::Optional<llvm::StringRef> filename)
    : ctx(ctx), name(ns_name) {
  if (filename.hasValue()) {
    this->filename.emplace(filename.getValue().str());
  }
};

void Namespace::enqueueError(llvm::StringRef e) {
  ctx.diagEngine->enqueueError(e);
}

exprs::Ast &Namespace::getTree() { return this->tree; }

mlir::LogicalResult Namespace::expandTree(exprs::Ast &ast) {

  if (ctx.getTargetPhase() == CompilationPhase::Parse) {
    // we just want the raw AST
    this->tree.insert(this->tree.end(), ast.begin(), ast.end());
    return mlir::success();
  }

  // Run the semantic analyer on the ast and then if everything
  // is ok expand the currnt tree by the semantically correct ast.
  auto maybeAst = reader::analyze(ctx, ast);

  if (!maybeAst) {
    enqueueError("Semantic analysis failed!");
    return mlir::failure();
  }

  auto semanticAst = std::move(maybeAst.getValue());
  this->tree.insert(this->tree.end(), semanticAst.begin(), semanticAst.end());
  return mlir::success();
}

uint Namespace::nextFnCounter() { return fn_counter++; };

SereneContext &Namespace::getContext() { return this->ctx; };

MaybeModuleOp Namespace::generate() {
  mlir::OpBuilder builder(&ctx.mlirContext);
  // TODO: Fix the unknown location by pointing to the `ns` form
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc(),
                                       llvm::Optional<llvm::StringRef>(name));

  // Walk the AST and call the `generateIR` function of each node.
  // Since nodes will have access to the a reference of the
  // namespace they can use the builder and keep adding more
  // operations to the module via the builder
  for (auto &x : getTree()) {
    x->generateIR(*this, module);
  }

  if (mlir::failed(mlir::verify(module))) {
    module.emitError("Can't verify the module");
    module.erase();
    return llvm::None;
  }

  if (mlir::failed(runPasses(module))) {
    // TODO: Report a proper error
    module.emitError("Failure in passes!");
    module.erase();
    return llvm::None;
  }

  return MaybeModuleOp(module);
}

mlir::LogicalResult Namespace::runPasses(mlir::ModuleOp &m) {
  return ctx.pm.run(m);
};

void Namespace::dump() {
  llvm::outs() << "\nMLIR: \n";
  auto maybeModuleOp = generate();

  if (!maybeModuleOp) {
    llvm::errs() << "Failed to generate the IR.\n";
    return;
  }

  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo();

  maybeModuleOp.getValue()->print(llvm::outs(), flags);
};

MaybeModule Namespace::compileToLLVM() {
  auto maybeModule = generate();

  if (!maybeModule) {
    NAMESPACE_LOG("IR generation failed for '" << name << "'");
    return llvm::None;
  }

  if (ctx.getTargetPhase() >= CompilationPhase::IR) {
    mlir::ModuleOp module = maybeModule.getValue().get();
    return MaybeModule(::serene::slir::compileToLLVMIR(ctx, module));
  }

  return llvm::None;
};

Namespace::~Namespace(){};

std::shared_ptr<Namespace>
makeNamespace(SereneContext &ctx, llvm::StringRef name,
              llvm::Optional<llvm::StringRef> filename, bool setCurrent) {
  auto nsPtr = std::make_shared<Namespace>(ctx, name, filename);
  ctx.insertNS(nsPtr);
  if (setCurrent) {
    if (!ctx.setCurrentNS(nsPtr->name)) {
      throw std::runtime_error("Couldn't set the current NS");
    }
  }
  return nsPtr;
};
} // namespace serene
