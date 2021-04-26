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

#ifndef NAMESPACE_H
#define NAMESPACE_H

#include "serene/environment.h"
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Module.h>
#include <mlir/Support/LogicalResult.h>
#include <string>

#define NAMESPACE_LOG(...)                                                     \
  DEBUG_WITH_TYPE("NAMESPACE", llvm::dbgs() << __VA_ARGS__ << "\n");

namespace serene {
namespace exprs {
class Expression;
using Node = std::shared_ptr<Expression>;
using Ast = std::vector<Node>;
} // namespace exprs

class Namespace {
private:
  bool initialized = false;
  exprs::Ast tree;

public:
  mlir::StringRef name;
  llvm::Optional<llvm::StringRef> filename;

  /// The root environment of the namespace on the semantic analysis phase.
  /// Which is a mapping from names to AST nodes ( no evaluation ).
  Environment<llvm::StringRef, exprs::Node> semanticEnv;

  Namespace(llvm::StringRef ns_name, llvm::Optional<llvm::StringRef> filename);

  exprs::Ast &Tree();
  mlir::LogicalResult setTree(exprs::Ast &);

  ~Namespace();
};

} // namespace serene

#endif
