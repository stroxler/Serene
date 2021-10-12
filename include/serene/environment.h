/* -*- C++ -*-
 * Serene Programming Language
 *
 * Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
