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

#include "serene/sir/generator.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "serene/expr.hpp"
#include "serene/sir/dialect.hpp"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace serene {
namespace sir {

mlir::ModuleOp Generator::generate() {
  for (auto x : ns->Tree()) {
    module.push_back(generate(x.get()));
  }

  return module;
};

mlir::Operation *Generator::generate(AExpr *x) {
  switch (x->getType()) {
  case SereneType::Number: {
    return generate(llvm::cast<Number>(x));
  }

  case SereneType::List: {
    return generate(llvm::cast<List>(x));
  }

  default: {
    return builder.create<ValueOp>(builder.getUnknownLoc(), (uint64_t)3);
  }
  }
};

mlir::Operation *Generator::generate(List *l) {
  // auto first = l->at(0);

  // if (!first) {
  //   // Empty list.
  //   // TODO: Return Nil or empty list.

  //   // Just for now.
  //   return builder.create<ValueOp>(builder.getUnknownLoc(), (uint64_t)0);
  // }

  // if (first->get()->getType() == SereneType::Symbol) {
  //   auto fnNameSymbol = llvm::dyn_cast<Symbol>(first->get());

  //   switch (fnNameSymbol->getName()) {
  //   case "def": {
  //     if (l->count() != 3) {
  //       llvm_unreachable("'def' form needs exactly 2 arguments.");
  //     }

  //     auto nameSymbol = llvm::dyn_cast<Symbol>(l->at(1).getValue().get());

  //     if (!nameSymbol) {
  //       llvm_unreachable("The first element of 'def' has to be a symbol.");
  //     }
  //     auto value = l->at(2).getValue().get();
  //     auto fn = generate(value);
  //     auto loc(value->location->start);
  //     // Define a function

  //     ns.insert_symbol(nameSymbol->getName(), llvm::cast<llvm::Value>(fn));
  //     // This is a generic function, the return type will be inferred later.
  //     // Arguments type are uniformly unranked tensors.
  //     break;
  //   }
  //   default: {
  //   }
  //   }
  // }
  // auto rest = l->from(1);

  // for (auto x : *rest) {
  //   generate(x.get());
  // }

  return builder.create<ValueOp>(builder.getUnknownLoc(), (uint64_t)0);
};

mlir::Operation *Generator::generate(Number *x) {
  return builder.create<ValueOp>(builder.getUnknownLoc(), x->toI64());
};

/**
 * Convert a Serene location to MLIR FileLineLoc Location
 */
::mlir::Location Generator::toMLIRLocation(serene::reader::Location *loc) {
  auto file = ns->filename;
  std::string filename{file.getValueOr("REPL")};

  return mlir::FileLineColLoc::get(builder.getIdentifier(filename), loc->line,
                                   loc->col);
}

Generator::~Generator(){};
} // namespace sir

} // namespace serene
