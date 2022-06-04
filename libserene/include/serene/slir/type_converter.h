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

#ifndef SERENE_SLIR_TYPE_CONVERTER_H
#define SERENE_SLIR_TYPE_CONVERTER_H

#include "serene/config.h"
#include "serene/context.h"
#include "serene/export.h"
#include "serene/slir/dialect.h"
#include "serene/slir/ops.h"
#include "serene/slir/types.h"
#include "serene/utils.h"

#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

namespace serene::slir {
class SymbolType;
class PtrType;

/// Returns the LLVM pointer type to the type `T` that is Serene pointer
/// `p` is pointing to.
mlir::Type getPtrTypeinLLVM(mlir::MLIRContext &ctx, PtrType p);

/// Returns the conversion result of converting Serene String type
/// to LLVM dialect
mlir::Type getStringTypeinLLVM(mlir::MLIRContext &ctx);

/// Returns the conversoin result of converting Serene Symbol type
/// to LLVM dialect
mlir::Type getSymbolTypeinLLVM(mlir::MLIRContext &ctx);

class TypeConverter : public mlir::TypeConverter {
  mlir::MLIRContext &ctx;

  using mlir::TypeConverter::convertType;

  using MaybeType   = llvm::Optional<mlir::Type>;
  using MaybeValue  = llvm::Optional<mlir::Value>;
  using ConverterFn = std::function<MaybeType(mlir::Type)>;

  std::function<MaybeValue(mlir::OpBuilder &, SymbolType, mlir::ValueRange,
                           mlir::Location)>

  materializeSymbolFn() {
    return [&](mlir::OpBuilder &builder, SymbolType type,
               mlir::ValueRange values, mlir::Location loc) -> MaybeValue {
      auto targetType = convertType(type);
      auto ret        = builder.create<ConvertOp>(loc, targetType, values[0]);
      llvm::outs() << "RR: " << ret << "\n";
      return ret.getResult();
    };
  };

  /// This method will be called via `convertType` to convert Serene types
  /// to llvm dialect types
  ConverterFn convertSereneTypes();

public:
  TypeConverter(mlir::MLIRContext &ctx) : ctx(ctx) {
    addConversion([](mlir::Type type) { return type; });
    addConversion(convertSereneTypes());
    addArgumentMaterialization(materializeSymbolFn());
    addTargetMaterialization(materializeSymbolFn());
  }
};
} // namespace serene::slir

#endif
