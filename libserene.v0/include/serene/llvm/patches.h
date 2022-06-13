/* -*- C++ -*-
 * Serene Programming Language
 *
 * Copyright (c) 2019-2022 Sameer Rahmani <lxsameer@gnu.org>
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

#ifndef LLVM_PATCHES_H
#define LLVM_PATCHES_H

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Hashing.h>

namespace llvm {

// Our specialization of DensMapInfo for string type. This will allow use to use
// string
template <>
struct DenseMapInfo<std::string> {
  static inline std::string getEmptyKey() { return ""; }

  static inline std::string getTombstoneKey() {
    // Maybe we need to use something else beside strings ????
    return "0TOMBED";
  }

  static unsigned getHashValue(const std::string &Val) {
    assert(Val != getEmptyKey() && "Cannot hash the empty key!");
    assert(Val != getTombstoneKey() && "Cannot hash the tombstone key!");
    return (unsigned)(llvm::hash_value(Val));
  }

  static bool isEqual(const std::string &LHS, const std::string &RHS) {
    if (RHS == getEmptyKey()) {
      return LHS == getEmptyKey();
    }
    if (RHS == getTombstoneKey()) {
      return LHS == getTombstoneKey();
    }
    return LHS == RHS;
  }
};

} // namespace llvm

#endif
