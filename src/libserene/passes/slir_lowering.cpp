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

#include "serene/passes.h"
#include "serene/slir/dialect.h"
#include "serene/utils.h"

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstdint>

namespace serene::passes {

// ----------------------------------------------------------------------------
// ValueOp lowering to constant op
struct ValueOpLowering : public mlir::OpRewritePattern<serene::slir::ValueOp> {
  using OpRewritePattern<serene::slir::ValueOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(serene::slir::ValueOp op,
                  mlir::PatternRewriter &rewriter) const final;
};

mlir::LogicalResult
ValueOpLowering::matchAndRewrite(serene::slir::ValueOp op,
                                 mlir::PatternRewriter &rewriter) const {
  auto value         = op.value();
  mlir::Location loc = op.getLoc();

  llvm::SmallVector<mlir::Type, 1> arg_types(0);
  auto func_type = rewriter.getFunctionType(arg_types, rewriter.getI64Type());
  // TODO: use a mechanism to generate unique names
  auto fn = rewriter.create<mlir::FuncOp>(loc, "randomname", func_type);

  auto *entryBlock = fn.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  // Since we only support i64 at the moment we use ConstantOp
  auto retVal = rewriter
                    .create<mlir::arith::ConstantIntOp>(loc, (int64_t)value,
                                                        rewriter.getI64Type())
                    .getResult();

  UNUSED(rewriter.create<mlir::ReturnOp>(loc, retVal));

  fn.setPrivate();

  // Erase the original ValueOP
  rewriter.eraseOp(op);
  return mlir::success();
}

// ----------------------------------------------------------------------------
// Fn lowering pattern
struct FnOpLowering : public mlir::OpRewritePattern<serene::slir::FnOp> {
  using OpRewritePattern<serene::slir::FnOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(serene::slir::FnOp op,
                  mlir::PatternRewriter &rewriter) const final;
};

mlir::LogicalResult
FnOpLowering::matchAndRewrite(serene::slir::FnOp op,
                              mlir::PatternRewriter &rewriter) const {
  auto args          = op.args();
  auto name          = op.name();
  auto isPublic      = op.sym_visibility().getValueOr("public") == "public";
  mlir::Location loc = op.getLoc();

  llvm::SmallVector<mlir::Type, 4> arg_types;

  for (const auto &arg : args) {
    auto attr = arg.getValue().dyn_cast<mlir::TypeAttr>();

    if (!attr) {
      op.emitError("It's not a type attr");
      return mlir::failure();
    }
    arg_types.push_back(attr.getValue());
  }

  auto func_type = rewriter.getFunctionType(arg_types, rewriter.getI64Type());
  auto fn        = rewriter.create<mlir::FuncOp>(loc, name, func_type);

  auto *entryBlock = fn.addEntryBlock();

  rewriter.setInsertionPointToStart(entryBlock);

  auto retVal = rewriter
                    .create<mlir::arith::ConstantIntOp>(loc, (int64_t)3,
                                                        rewriter.getI64Type())
                    .getResult();

  rewriter.create<mlir::ReturnOp>(loc, retVal);

  if (!isPublic) {
    fn.setPrivate();
  }

  rewriter.eraseOp(op);
  return mlir::success();
}

// ----------------------------------------------------------------------------
// SLIR lowering pass
// This Pass will lower SLIR to MLIR's standard dialect.
struct SLIRToMLIRPass
    : public mlir::PassWrapper<SLIRToMLIRPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override;
  void runOnOperation() final;
  void runOnModule();
  mlir::ModuleOp getModule();
};

// Mark what dialects we need for this pass. It's basically translate to what
// dialects do we want to lower to
void SLIRToMLIRPass::getDependentDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<mlir::StandardOpsDialect, mlir::arith::ArithmeticDialect>();
};

/// Return the current function being transformed.
mlir::ModuleOp SLIRToMLIRPass::getModule() { return this->getOperation(); }

void SLIRToMLIRPass::runOnOperation() { runOnModule(); }

void SLIRToMLIRPass::runOnModule() {

  auto module = getModule();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  mlir::ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to the `Standard` dialects.
  target.addLegalDialect<mlir::StandardOpsDialect>();
  target.addLegalDialect<mlir::arith::ArithmeticDialect>();

  // We also define the SLIR dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted.
  target.addIllegalDialect<serene::slir::SereneDialect>();

  // Mark operations that are LEGAL for this pass. It means that we don't lower
  // them is this pass but we will in another pass. So we don't want to get
  // an error since we are not lowering them.
  // target.addLegalOp<serene::slir::PrintOp>();
  target.addLegalOp<mlir::FuncOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the SLIR operations.
  mlir::RewritePatternSet patterns(&getContext());

  // Pattern to lower ValueOp and FnOp
  patterns.add<ValueOpLowering, FnOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createSLIRLowerToMLIRPass() {
  return std::make_unique<SLIRToMLIRPass>();
};
} // namespace serene::passes
