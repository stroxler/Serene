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
#ifndef SERENE_SLIR_DIALECT_H
#define SERENE_SLIR_DIALECT_H

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

// Include the auto-generated header file containing the declaration of the
// serene's dialect.
#include "serene/slir/dialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "serene/slir/types.h.inc"
// Include the auto-generated header file containing the declarations of the
// serene's operations.
// for more on GET_OP_CLASSES: https://mlir.llvm.org/docs/OpDefinitions/
#define GET_OP_CLASSES
#include "serene/slir/ops.h.inc"

#endif // SERENE_SLIR_DIALECT_H
