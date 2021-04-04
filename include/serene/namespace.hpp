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

#ifndef NAMESPACE_H
#define NAMESPACE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "serene/expr.hpp"
#include "serene/llvm/IR/Value.h"
#include "serene/logger.hpp"
#include "llvm/ADT/DenseMap.h"
#include <llvm/IR/Module.h>
#include <string>

#if defined(ENABLE_LOG) || defined(ENABLE_NAMESPACE_LOG)
#define NAMESPACE_LOG(...) __LOG("NAMESPACE", __VA_ARGS__);
#else
#define NAMESPACE_LOG(...) ;
#endif

using ScopeMap = llvm::DenseMap<llvm::StringRef, mlir::Value>;
using PairT = std::pair<llvm::StringRef, mlir::Value>;

namespace serene {
class AExpr;

class Namespace {
private:
  ast_tree tree{};
  bool initialized = false;

  ScopeMap rootScope;

public:
  llvm::Optional<llvm::StringRef> filename;
  mlir::StringRef name;

  Namespace(llvm::StringRef ns_name, llvm::Optional<llvm::StringRef> filename);

  ast_tree &Tree();
  mlir::LogicalResult setTree(ast_tree);
  // TODO: Fix it to return llvm::Optional<mlir::Value> instead
  llvm::Optional<mlir::Value> lookup(llvm::StringRef name);
  mlir::LogicalResult insert_symbol(llvm::StringRef name, mlir::Value v);

  void print_scope();
  ~Namespace();
};

} // namespace serene

#endif
