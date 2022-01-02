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
#include "serene/semantics.h"
#include "serene/slir/slir.h"

#include <serene/export.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Support/LogicalResult.h>

#include <memory>
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

  // Create the root environment
  createEnv(nullptr);
};

void Namespace::enqueueError(llvm::StringRef e) const {
  ctx.diagEngine->enqueueError(e);
}

SemanticEnv &Namespace::createEnv(SemanticEnv *parent) {
  auto env = std::make_unique<SemanticEnv>(parent);
  environments.push_back(std::move(env));

  return *environments.back();
};

SemanticEnv &Namespace::getRootEnv() {
  assert(!environments.empty() && "Root env is not created!");

  return *environments.front();
};

mlir::LogicalResult Namespace::define(std::string &name, exprs::Node &node) {
  auto &rootEnv = getRootEnv();

  if (failed(rootEnv.insert_symbol(name, node))) {
    return mlir::failure();
  }

  symbolList.push_back(name);
  return mlir::success();
}

exprs::Ast &Namespace::getTree() { return this->tree; }
errors::OptionalErrors Namespace::addTree(exprs::Ast &ast) {

  // TODO: Remove the parse phase
  if (ctx.getTargetPhase() == CompilationPhase::Parse) {
    // we just want the raw AST
    this->tree.insert(this->tree.end(), ast.begin(), ast.end());
    return llvm::None;
  }
  auto &rootEnv = getRootEnv();

  auto state = semantics::makeAnalysisState(*this, rootEnv);
  // Run the semantic analyer on the ast and then if everything
  // is ok add the form to the tree and forms
  auto maybeForm = semantics::analyze(*state, ast);

  if (!maybeForm) {
    return maybeForm.getError();
  }

  auto semanticAst = std::move(maybeForm.getValue());
  this->tree.insert(this->tree.end(), semanticAst.begin(), semanticAst.end());

  return llvm::None;
}

uint Namespace::nextFnCounter() { return fn_counter++; };

SereneContext &Namespace::getContext() { return this->ctx; };

MaybeModuleOp Namespace::generate(unsigned offset) {
  mlir::OpBuilder builder(&ctx.mlirContext);
  // TODO: Fix the unknown location by pointing to the `ns` form
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc(),
                                       llvm::Optional<llvm::StringRef>(name));

  auto treeSize = getTree().size();

  // Walk the AST and call the `generateIR` function of each node.
  // Since nodes will have access to the a reference of the
  // namespace they can use the builder and keep adding more
  // operations to the module via the builder
  for (unsigned i = offset; i < treeSize; ++i) {
    auto &node = getTree()[i];
    node->generateIR(*this, module);
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
    return ::serene::slir::compileToLLVMIR(ctx, module);
  }

  return llvm::None;
};

MaybeModule Namespace::compileToLLVMFromOffset(unsigned offset) {
  auto maybeModule = generate(offset);

  if (!maybeModule) {
    NAMESPACE_LOG("IR generation failed for '" << name << "'");
    return llvm::None;
  }

  if (ctx.getTargetPhase() >= CompilationPhase::IR) {
    mlir::ModuleOp module = maybeModule.getValue().get();
    return ::serene::slir::compileToLLVMIR(ctx, module);
  }

  return llvm::None;
};

NSPtr Namespace::make(SereneContext &ctx, llvm::StringRef name,
                      llvm::Optional<llvm::StringRef> filename) {
  return std::make_shared<Namespace>(ctx, name, filename);
};

Namespace::~Namespace(){};

} // namespace serene
