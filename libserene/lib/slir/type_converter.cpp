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

#include "serene/slir/type_converter.h"

namespace ll = mlir::LLVM;

namespace serene {
namespace slir {

mlir::Type getStringTypeinLLVM(mlir::MLIRContext &ctx) {
  auto stringStruct =
      ll::LLVMStructType::getIdentified(&ctx, "serene.core.string");

  mlir::SmallVector<mlir::Type, 4> subtypes;

  subtypes.push_back(
      ll::LLVMPointerType::get(mlir::IntegerType::get(&ctx, I8_SIZE)));
  subtypes.push_back(mlir::IntegerType::get(&ctx, I32_SIZE));
  // subtypes.push_back(mlir::IntegerType::get(&ctx, I32_SIZE));
  // subtypes.push_back(mlir::IntegerType::get(&ctx, I32_SIZE));

  (void)stringStruct.setBody(subtypes, false);

  return ll::LLVMPointerType::get(stringStruct);
};

mlir::Type getSymbolTypeinLLVM(mlir::MLIRContext &ctx) {
  auto symbolStruct =
      ll::LLVMStructType::getIdentified(&ctx, "serene.core.symbol");

  auto strType = getStringTypeinLLVM(ctx);
  llvm::SmallVector<mlir::Type, 2> strings{strType, strType};

  // We discard the result becasue if the body was already set it means
  // that we're ok and the struct already exists
  (void)symbolStruct.setBody(strings, false);
  return ll::LLVMPointerType::get(symbolStruct);
};

TypeConverter::ConverterFn TypeConverter::convertSereneTypes() {
  return [&](mlir::Type type) -> MaybeType {
    if (type.isa<StringType>()) {
      return getStringTypeinLLVM(ctx);
    }

    if (type.isa<SymbolType>()) {
      return getSymbolTypeinLLVM(ctx);
    }

    return llvm::None;
  };
}
} // namespace slir
} // namespace serene
