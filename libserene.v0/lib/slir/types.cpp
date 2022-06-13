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

#include "serene/slir/types.h"

#include "serene/slir/dialect.h"

#define GET_ATTRDEF_CLASSES
#include "serene/slir/attrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "serene/slir/types.cpp.inc"

namespace serene::slir {

PtrType PtrType::get(mlir::MLIRContext *context, unsigned addressSpace) {
  return Base::get(context, mlir::Type(), addressSpace);
}

PtrType PtrType::get(mlir::Type pointee, unsigned addressSpace) {
  return Base::get(pointee.getContext(), pointee, addressSpace);
}

bool PtrType::isOpaque() const { return !getImpl()->pointeeType; }

void SereneDialect::registerType() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "serene/slir/attrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "serene/slir/types.cpp.inc"
      >();
};

} // namespace serene::slir
