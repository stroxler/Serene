/*
 * Serene programming language.
 *
 *  Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Identifier.h"
#include "serene/slir/dialect.h"

#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/Builders.h>
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
