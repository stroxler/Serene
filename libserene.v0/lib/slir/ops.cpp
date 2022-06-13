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

#include "serene/slir/ops.h"

#include "serene/slir/dialect.h"
#include "serene/slir/types.h"
#include "serene/utils.h"

#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OperationSupport.h>

#define GET_OP_CLASSES
#include "serene/slir/ops.cpp.inc"

namespace serene::slir {

mlir::DataLayoutSpecInterface NsOp::getDataLayoutSpec() {
  // Take the first and only (if present) attribute that implements the
  // interface. This needs a linear search, but is called only once per data
  // layout object construction that is used for repeated queries.
  for (mlir::NamedAttribute attr : getOperation()->getAttrs()) {
    if (auto spec = attr.getValue().dyn_cast<mlir::DataLayoutSpecInterface>()) {
      return spec;
    }
  }
  return {};
}

mlir::OpFoldResult SymbolOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  UNUSED(operands);
  return value();
};

mlir::OpFoldResult ValueOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  UNUSED(operands);
  return value();
};
} // namespace serene::slir
