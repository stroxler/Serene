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

#include "serene/namespace.h"
#include "serene/exprs/expression.h"
#include "serene/llvm/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include <fmt/core.h>
#include <string>

using namespace std;
using namespace llvm;

namespace serene {

Namespace::Namespace(llvm::StringRef ns_name,
                     llvm::Optional<llvm::StringRef> filename)
    : name(ns_name) {
  if (filename.hasValue()) {
    this->filename.emplace(filename.getValue().str());
  }
};

exprs::Ast &Namespace::getTree() { return this->tree; }

mlir::LogicalResult Namespace::setTree(exprs::Ast &t) {
  if (initialized) {
    return mlir::failure();
  }
  this->tree = std::move(t);
  this->initialized = true;
  return mlir::success();
}

std::shared_ptr<Namespace>
makeNamespace(SereneContext &ctx, llvm::StringRef name,
              llvm::Optional<llvm::StringRef> filename, bool setCurrent) {
  auto nsPtr = std::make_shared<Namespace>(name, filename);
  ctx.insertNS(nsPtr);
  if (setCurrent) {
    assert(ctx.setCurrentNS(nsPtr->name) && "Couldn't set the current NS");
  }
  return nsPtr;
};

Namespace::~Namespace() {}

} // namespace serene
