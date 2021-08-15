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
#include "serene/slir/slir.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <stdexcept>
#include <string>

using namespace std;
using namespace llvm;

namespace serene {

Namespace::Namespace(SereneContext &ctx, llvm::StringRef ns_name,
                     llvm::Optional<llvm::StringRef> filename)
    : ctx(ctx), name(ns_name)

{
  if (filename.hasValue()) {
    this->filename.emplace(filename.getValue().str());
  }
};

exprs::Ast &Namespace::getTree() { return this->tree; }

mlir::LogicalResult Namespace::setTree(exprs::Ast &t) {
  if (initialized) {
    return mlir::failure();
  }
  this->tree        = std::move(t);
  this->initialized = true;
  return mlir::success();
}

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

uint Namespace::nextFnCounter() { return fn_counter++; };

SereneContext &Namespace::getContext() { return this->ctx; };

MaybeModuleOp Namespace::generate() {
  mlir::OpBuilder builder(&ctx.mlirContext);
  // TODO: Fix the unknown location by pointing to the `ns` form
  // TODO: We need to call `erase` method of module somewhere to clean it up
  //       maybe use a unique ptr?
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc(), name);

  for (auto &x : getTree()) {
    x->generateIR(*this, module);
  }

  if (mlir::failed(runPasses(module))) {
    // TODO: throw a proper errer
    module.emitError("Failure in passes!");
    return MaybeModuleOp::error(true);
  }

  return MaybeModuleOp::success(module);
}

mlir::LogicalResult Namespace::runPasses(mlir::ModuleOp &m) {
  return ctx.pm.run(m);
};

void Namespace::dump() {
  llvm::outs() << "\nMLIR: \n";
  auto maybeModuleOp = generate();

  if (!maybeModuleOp) {
    llvm::outs() << "Failed to generate the IR.\n";
    return;
  }

  maybeModuleOp.getValue().dump();
};

MaybeModule Namespace::compileToLLVM() {
  auto m = generate();

  if (!m) {
    NAMESPACE_LOG("IR generation failed for '" << name << "'");
    return MaybeModule::error(true);
  }

  if (ctx.getTargetPhase() >= CompilationPhase::IR) {
    return MaybeModule::success(
        ::serene::slir::compileToLLVMIR(ctx, m.getValue()));
  }

  return MaybeModule::error(true);
};

Namespace::~Namespace(){};

} // namespace serene
