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

#ifndef SERENE_ENVIRONMENT_H
#define SERENE_ENVIRONMENT_H

#include "serene/llvm/patches.h"

#include <llvm/ADT/DenseMap.h>
#include <mlir/Support/LogicalResult.h>

namespace serene {

/// This class represents a classic lisp environment (or scope) that holds the
/// bindings from type `K` to type `V`. For example an environment of symbols
/// to expressions would be `Environment<Symbol, Node>`
template <typename K, typename V>
class Environment {

  Environment<K, V> *parent;

  // The actual bindings storage
  llvm::DenseMap<K, V> pairs;

public:
  Environment() : parent(nullptr) {}
  Environment(Environment *parent) : parent(parent){};

  /// Look up the given `key` in the environment and return it.
  llvm::Optional<V> lookup(K key) {
    if (auto value = pairs.lookup(key)) {
      return value;
    }

    if (parent) {
      return parent->lookup(key);
    }

    return llvm::None;
  };

  /// Insert the given `key` with the given `value` into the storage. This
  /// operation will shadow an aleady exist `key` in the parent environment
  mlir::LogicalResult insert_symbol(K key, V value) {
    pairs.insert(std::pair<K, V>(key, value));
    return mlir::success();
  };
};

} // namespace serene

#endif
