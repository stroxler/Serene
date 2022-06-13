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

#ifndef SERENE_PASSES_TD
#define SERENE_PASSES_TD

include "mlir/Pass/PassBase.td"


// TODO: When we add and utilize namespace operations we need
//       use NsOp instead of the ModuleOp for root of lowering
def LowerSLIR : Pass<"lower-slir", "mlir::ModuleOp"> {
  let summary = "Lowers the SLIR dialect to LLVM IR indirectly";

  let description = [{
      Lowers the SLIR dialect to LLVM IR indirectly meaning
      that it will lower SLIR to other dialects that in turn will
      eventually lower to llvm ir.
  }];

  // This is how we can create the default instance of this pass.
  // via `createLowerSymbol` function
  let constructor = "serene::passes::createLowerSLIR()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::arith::ArithmeticDialect"];

}

#endif // SERENE_PASSES_TD
