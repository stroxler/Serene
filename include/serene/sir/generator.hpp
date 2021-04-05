/**
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
#ifndef GENERATOR_H
#define GENERATOR_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/MLIRContext.h"
#include "serene/expr.hpp"
#include "serene/list.hpp"
#include "serene/namespace.hpp"
#include "serene/number.hpp"
#include "serene/symbol.hpp"
#include "llvm/ADT/ScopedHashTable.h"
#include <atomic>
#include <memory>
#include <utility>

namespace serene {
namespace sir {

using FnIdPair = std::pair<mlir::Identifier, mlir::FuncOp>;

class Generator {
private:
  ::mlir::OpBuilder builder;
  ::mlir::ModuleOp module;
  std::unique_ptr<::serene::Namespace> ns;
  std::atomic_int anonymousFnCounter{1};
  llvm::DenseMap<mlir::Identifier, mlir::FuncOp> anonymousFunctions;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

  // TODO: Should we use builder here? maybe there is a better option
  ::mlir::Location toMLIRLocation(serene::reader::Location *);

  // mlir::FuncOp generateFn(serene::reader::Location, std::string, List *,
  //                         List *);

public:
  Generator(mlir::MLIRContext &context, ::serene::Namespace *ns)
      : builder(&context),
        module(mlir::ModuleOp::create(builder.getUnknownLoc(), ns->name)) {
    this->ns.reset(ns);
  }

  mlir::Operation *generate(Number *);
  mlir::Operation *generate(AExpr *);
  mlir::Value generate(List *);
  mlir::ModuleOp generate();
  ~Generator();
};

} // namespace sir

} // namespace serene

#endif
