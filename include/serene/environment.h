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

#include "mlir/Support/LogicalResult.h"
#include "serene/llvm/patches.h"

#include "llvm/ADT/DenseMap.h"

namespace serene {

template <typename K, typename V>
class Environment {
  Environment<K, V> *parent;
  llvm::DenseMap<K, V> pairs;

public:
  Environment() : parent(nullptr) {}
  Environment(Environment *parent) : parent(parent){};

  llvm::Optional<V> lookup(K key) {
    if (auto value = pairs.lookup(key)) {
      return value;
    }

    if (parent) {
      return parent->lookup(key);
    }

    return llvm::None;
  };

  mlir::LogicalResult insert_symbol(K key, V value) {
    pairs.insert(std::pair<K, V>(key, value));
    return mlir::success();
  };
};

} // namespace serene

#endif
