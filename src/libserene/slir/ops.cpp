/*
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

#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Identifier.h>
#include <mlir/IR/OperationSupport.h>

namespace serene::slir {
// void FnOp::build(mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState,
//                  llvm::StringRef name, mlir::FunctionType type,
//                  llvm::ArrayRef<mlir::NamedAttribute> attrs,
//                  llvm::ArrayRef<mlir::DictionaryAttr> argAttrs) {
//   odsState.addAttribute("name", odsBuilder.getStringAttr(name));
//   llvm::SmallVector<mlir::NamedAttribute> args;

//   for (unsigned int i = 0; i <= type.getNumInputs(); i++) {
//     odsState.addAttribute(attrName, mlir::TypeAttr::get(type.getResult(i)));
//     args.push_back(odsBuilder.getNamedAttr(,
//     mlir::IntegerAttr::get(odsBuilder.getI64Type(), type.getNumInputs())));
//   }

//   odsState.addAttribute("xxx",
//   mlir::DictionaryAttr::get(odsBuilder.getContext(), p));
//   odsState.addAttribute("input_count",
//   mlir::IntegerAttr::get(odsBuilder.getI64Type(),
//                                                               type.getNumInputs()));
//   // if (sym_visibility) {
//   // odsState.addAttribute("sym_visibility", sym_visibility);
//   // }
//   (void)odsState.addRegion();
//   odsState.addTypes(type);
// };

} // namespace serene::slir
