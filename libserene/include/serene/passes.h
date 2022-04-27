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

#ifndef SERENE_PASSES_H
#define SERENE_PASSES_H

#include "serene/export.h"

#include <mlir/Pass/Pass.h>

namespace serene::passes {

/// Return a pass to lower the serene.symbol op
SERENE_EXPORT std::unique_ptr<mlir::Pass> createLowerSymbol();

SERENE_EXPORT void registerAllPasses();
/// Return a pass to convert SLIR dialect to built-in dialects
/// of MLIR.
std::unique_ptr<mlir::Pass> createSLIRLowerToMLIRPass();

/// Return a pass to convert different dialects of MLIR to LLVM dialect.
std::unique_ptr<mlir::Pass> createSLIRLowerToLLVMDialectPass();

} // namespace serene::passes

#endif
