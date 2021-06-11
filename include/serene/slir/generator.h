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

#ifndef GENERATOR_H
#define GENERATOR_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Identifier.h"
#include "serene/context.h"
#include "serene/exprs/expression.h"
#include "serene/exprs/list.h"
#include "serene/exprs/number.h"
#include "serene/exprs/symbol.h"
#include "serene/namespace.h"
#include "llvm/ADT/ScopedHashTable.h"
#include <memory>
#include <utility>

namespace serene::slir {

class Generator {
private:
  serene::SereneContext &ctx;
  mlir::OpBuilder builder;
  mlir::ModuleOp module;
  std::shared_ptr<serene::Namespace> ns;

  // TODO: Should we use builder here? maybe there is a better option
  mlir::Location toMLIRLocation(serene::reader::Location &);

public:
  Generator(serene::SereneContext &ctx, llvm::StringRef ns_name)
      : ctx(ctx), builder(&ctx.mlirContext),
        module(mlir::ModuleOp::create(builder.getUnknownLoc(), ns_name)) {
    this->ns = ctx.getNS(ns_name);
  };

  void generate(exprs::Number &);
  mlir::Operation *generate(exprs::Expression *);
  mlir::Value generate(exprs::List *);
  mlir::ModuleOp generate();

  serene::Namespace &getNs();
  ~Generator();
};

} // namespace serene::slir

#endif
