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

// SereneSymbol is the Symbil type in SLIT that represents an expr::Sympol

#ifndef SERENE_SLIR_SYMBOL_H
#define SERENE_SLIR_SYMBOL_H

#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/TypeSupport.h>

namespace serene {
namespace slir {
namespace detail {
/// This is the storage class for the `SymbolType` (SereneSymbol in ODS)
// class SymbolTypeStorage : public mlir::TypeStorage {
//   using KeyTy = std::string;

//   SymbolTypeStorage(std::string &ns, std::string &name) : ns(ns), name(name)
//   {} SymbolTypeStorage(const KeyTy &k) {

//     auto partDelimiter = k.find('/');
//     if (partDelimiter == std::string::npos) {
//       llvm::llvm_unreachable_internal("SLIR symbol has to have NS");
//     } else {
//       name = k.substr(partDelimiter + 1, k.size());
//       ns   = k.substr(0, partDelimiter);
//     }
//   }

//   /// The hash key for this storage is a pair of the integer and type params.

//   /// Define the comparison function for the key type.
//   bool operator==(const KeyTy &key) const {
//     // TODO: Use formatv to concat strings
//     return key == ns + "/" + name;
//   }

//   static llvm::hash_code hashKey(const KeyTy &key) {
//     return llvm::hash_combine(key);
//   }

//   /// Define a construction function for the key type.
//   /// Note: This isn't necessary because KeyTy can be directly constructed
//   with
//   /// the given parameters.
//   static KeyTy getKey(std::string &ns, std::string &name) {
//     // TODO: Use formatv to concat strings
//     return KeyTy(ns + "/" + name);
//   }

//   /// Define a construction method for creating a new instance of this
//   storage. static SymbolTypeStorage *construct(mlir::TypeStorageAllocator
//   &allocator,
//                                       const KeyTy &key) {
//     return new (allocator.allocate<SymbolTypeStorage>())
//     SymbolTypeStorage(key);
//   }

//   std::string ns;
//   std::string name;
// };
}; // namespace detail
}; // namespace slir
}; // namespace serene
#endif
