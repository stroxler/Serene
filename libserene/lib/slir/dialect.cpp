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
#include "serene/slir/dialect.h"

#include "serene/slir/dialect.cpp.inc"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/MLIRContext.h>

namespace serene {
namespace slir {

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void SereneDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "serene/slir/ops.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "serene/slir/types.cpp.inc"
      >();
}

// static SymbolType parseSymbolType(mlir::MLIRContext &ctx,
//                                   mlir::AsmParser &parser) {
//   llvm::SMLoc loc = parser.getCurrentLocation();
//   if (parser.parseLess()) {
//     return SymbolType();
//   }
//   std::string fqsym;
//   if(parser.parseString(&fqsym)) {

//   }
// };

// /// Parses a type appearing inside another LLVM dialect-compatible type. This
// /// will try to parse any type in full form (including types with the `!llvm`
// /// prefix), and on failure fall back to parsing the short-hand version of
// the
// /// LLVM dialect types without the `!llvm` prefix.
// static mlir::Type dispatchParse(mlir::AsmParser &parser, bool allowAny =
// true) {
//   llvm::SMLoc keyLoc = parser.getCurrentLocation();

//   // Try parsing any MLIR type.
//   mlir::Type type;
//   mlir::OptionalParseResult result = parser.parseOptionalType(type);

//   if (result.hasValue()) {
//     if (failed(result.getValue())) {
//       return nullptr;
//     }

//     if (!allowAny) {
//       parser.emitError(keyLoc) << "unexpected type, expected keyword";
//       return nullptr;
//     }
//     return type;
//   }

//   // If no type found, fallback to the shorthand form.
//   llvm::StringRef key;

//   if (failed(parser.parseKeyword(&key))) {
//     return mlir::Type();
//   }

//   mlir::MLIRContext *ctx = parser.getContext();
//   return mlir::StringSwitch<mlir::function_ref<mlir::Type()>>(key)
//       .Case("symbol", [&] { return parseSymbolType(ctx, parser); })
//       .Default([&] {
//         parser.emitError(keyLoc) << "unknown LLVM type: " << key;
//         return mlir::Type();
//       })();
// }
// //.Case("struct", [&] { return parseStructType(parser); })
// /// Parse an instance of a type registered to the dialect.
// mlir::Type SereneDialect::parseType(mlir::DialectAsmParser &parser) const {
//   llvm::SMLoc loc = parser.getCurrentLocation();
//   mlir::Type type = dispatchParse(parser, /*allowAny=*/false);
//   if (!type) {
//     return type;
//   }
//   if (!isCompatibleOuterType(type)) {
//     parser.emitError(loc) << "unexpected type, expected keyword";
//     return nullptr;
//   }
//   return type;
// };

// // /// Print an instance of a type registered to the dialect.
// void SereneDialect::printType(
//     mlir::Type type, mlir::DialectAsmPrinter &printer) const override{};

} // namespace slir
} // namespace serene
#define GET_TYPEDEF_CLASSES
#include "serene/slir/types.cpp.inc"

#define GET_OP_CLASSES
#include "serene/slir/ops.cpp.inc"
