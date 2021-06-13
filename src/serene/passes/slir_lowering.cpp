/* -*- C++ -*-
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

#include "serene/passes/slir_lowering.h"

#include <mlir/IR/BuiltinOps.h>

namespace serene::passes {

mlir::LogicalResult
ValueOpLowering::matchAndRewrite(serene::slir::ValueOp op,
                                 mlir::PatternRewriter &rewriter) const {
  auto value         = op.value();
  mlir::Location loc = op.getLoc();

  // auto std_const =
  rewriter.create<mlir::ConstantIntOp>(loc, (int64_t)value,
                                       rewriter.getI64Type());
  // rewriter.replaceOpWithNewOp<typename OpTy>(Operation *op, Args &&args...)
  //  Replace this operation with the generated alloc.
  //  rewriter.replaceOp(op, alloc);
  rewriter.eraseOp(op);
  return mlir::success();
}

void SLIRToAffinePass::getDependentDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<mlir::AffineDialect, mlir::memref::MemRefDialect,
                  mlir::StandardOpsDialect>();
};

/// Return the current function being transformed.
mlir::ModuleOp SLIRToAffinePass::getModule() { return this->getOperation(); }

void SLIRToAffinePass::runOnOperation() { runOnModule(); }

void SLIRToAffinePass::runOnModule() {

  auto module = getModule();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  mlir::ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `MemRef` and `Standard` dialects.
  target.addLegalDialect<mlir::AffineDialect, mlir::memref::MemRefDialect,
                         mlir::StandardOpsDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as `legal`.
  target.addIllegalDialect<serene::slir::SereneDialect>();
  // target.addLegalOp<serene::slir::PrintOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<ValueOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
};

std::unique_ptr<mlir::Pass> createSLIRLowerToAffinePass() {
  return std::make_unique<SLIRToAffinePass>();
};
} // namespace serene::passes
