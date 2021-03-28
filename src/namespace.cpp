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

#include "serene/namespace.hpp"
#include "serene/expr.hpp"
#include "serene/llvm/IR/Value.h"
#include <fmt/core.h>
#include <string>

using namespace std;
using namespace llvm;

namespace serene {

ast_tree &Namespace::Tree() { return this->tree; }

mlir::Value Namespace::lookup(mlir::StringRef name) {
  if (auto value = scope.lookup(name)) {
    return value;
  }

  return nullptr;
};

mlir::LogicalResult Namespace::setTree(ast_tree t) {
  if (initialized) {
    return mlir::failure();
  }
  this->tree = t;
  this->initialized = true;
  return mlir::success();
}

mlir::LogicalResult Namespace::insert_symbol(mlir::StringRef name,
                                             mlir::Value v) {
  if (scope.count(name)) {
    return mlir::failure();
  }

  scope.insert(name, v);
  return mlir::success();
}

void Namespace::print_scope(){};

Namespace::~Namespace() {}

} // namespace serene
