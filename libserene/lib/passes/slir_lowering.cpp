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

#include "serene/context.h"
#include "serene/conventions.h"
#include "serene/passes.h"
#include "serene/slir/dialect.h"
#include "serene/slir/type_converter.h"
#include "serene/utils.h"

#include <serene/config.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/thread.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstdint>

namespace ll = mlir::LLVM;

namespace serene::passes {

static ll::GlobalOp getOrCreateInternalString(mlir::Location loc,
                                              mlir::OpBuilder &builder,
                                              llvm::StringRef name,
                                              llvm::StringRef value,
                                              mlir::ModuleOp module) {

  // Create the global at the entry of the module.
  ll::GlobalOp global;

  if (!(global = module.lookupSymbol<ll::GlobalOp>(name))) {
    mlir::OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());

    auto type = ll::LLVMArrayType::get(
        mlir::IntegerType::get(builder.getContext(), I8_SIZE), value.size());
    // TODO: Do we want link once ?
    global = builder.create<ll::GlobalOp>(loc, type, /*isConstant=*/true,
                                          ll::Linkage::Linkonce, name,
                                          builder.getStringAttr(value),
                                          /*alignment=*/0);
  }

  return global;
};

static mlir::Value getPtrToInternalString(mlir::OpBuilder &builder,
                                          ll::GlobalOp global) {
  auto loc = global.getLoc();
  auto I8  = mlir::IntegerType::get(builder.getContext(), I8_SIZE);
  // Get the pointer to the first character in the global string.
  mlir::Value globalPtr = builder.create<ll::AddressOfOp>(loc, global);
  mlir::Value cst0      = builder.create<ll::ConstantOp>(
      loc, mlir::IntegerType::get(builder.getContext(), I64_SIZE),
      builder.getIntegerAttr(builder.getIndexType(), 0));

  return builder.create<ll::GEPOp>(loc, ll::LLVMPointerType::get(I8), globalPtr,
                                   llvm::ArrayRef<mlir::Value>({cst0}));
};

static ll::GlobalOp getOrCreateString(mlir::Location loc,
                                      mlir::OpBuilder &builder,
                                      llvm::StringRef name,
                                      llvm::StringRef value, uint32_t len,
                                      mlir::ModuleOp module) {
  auto *ctx = builder.getContext();
  ll::GlobalOp global;

  if (!(global = module.lookupSymbol<ll::GlobalOp>(name))) {

    mlir::OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());

    mlir::Attribute initValue{};
    auto type = slir::getStringTypeinLLVM(*ctx);

    global = builder.create<ll::GlobalOp>(
        loc, type, /*isConstant=*/true, ll::Linkage::Linkonce, name, initValue);

    auto &gr    = global.getInitializerRegion();
    auto *block = builder.createBlock(&gr);

    builder.setInsertionPoint(block, block->begin());

    mlir::Value structInstant = builder.create<ll::UndefOp>(loc, type);

    auto strOp = getOrCreateInternalString(loc, builder, name, value, module);
    auto ptrToStr = getPtrToInternalString(builder, strOp);

    auto length = builder.create<ll::ConstantOp>(
        loc, mlir::IntegerType::get(ctx, I32_SIZE),
        builder.getI32IntegerAttr(len));

    // Setting the string pointer field
    structInstant = builder.create<ll::InsertValueOp>(
        loc, structInstant.getType(), structInstant, ptrToStr,
        builder.getI64ArrayAttr(0));

    // Setting the len field
    structInstant = builder.create<ll::InsertValueOp>(
        loc, structInstant.getType(), structInstant, length,
        builder.getI64ArrayAttr(1));

    builder.create<ll::ReturnOp>(loc, structInstant);
  }

  return global;
};

static ll::GlobalOp getOrCreateSymbol(mlir::Location loc,
                                      mlir::OpBuilder &builder,
                                      llvm::StringRef ns, llvm::StringRef name,
                                      mlir::ModuleOp module) {
  std::string fqName;
  ll::GlobalOp global;

  auto *ctx    = builder.getContext();
  auto symName = serene::mangleInternalSymName(fqName);

  makeFQSymbolName(ns, name, fqName);

  if (!(global = module.lookupSymbol<ll::GlobalOp>(symName))) {
    mlir::OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());

    mlir::Attribute initValue{};
    auto type = slir::getSymbolTypeinLLVM(*ctx);

    // We want to allow merging the strings representing the ns or name part
    // of the symbol with other modules to unify them.
    ll::Linkage linkage = ll::Linkage::Linkonce;

    global = builder.create<ll::GlobalOp>(loc, type, /*isConstant=*/true,
                                          linkage, symName, initValue);

    auto &gr    = global.getInitializerRegion();
    auto *block = builder.createBlock(&gr);

    builder.setInsertionPoint(block, block->begin());

    mlir::Value structInstant = builder.create<ll::UndefOp>(loc, type);

    // We want to use the mangled ns as the name of the constant that
    // holds the ns string
    auto mangledNSName = serene::mangleInternalStringName(ns);
    // The globalop that we want to use for the ns field
    auto nsField =
        getOrCreateString(loc, builder, mangledNSName, ns, ns.size(), module);
    auto ptrToNs = builder.create<ll::AddressOfOp>(loc, nsField);

    // We want to use the mangled 'name' as the name of the constant that
    // holds the 'name' string
    auto mangledName = serene::mangleInternalStringName(name);
    // The global op to use as the 'name' field
    auto nameField =
        getOrCreateString(loc, builder, mangledName, name, name.size(), module);
    auto ptrToName = builder.create<ll::AddressOfOp>(loc, nameField);

    // Setting the string pointer field
    structInstant = builder.create<ll::InsertValueOp>(
        loc, structInstant.getType(), structInstant, ptrToNs,
        builder.getI64ArrayAttr(0));

    // Setting the len field
    structInstant = builder.create<ll::InsertValueOp>(
        loc, structInstant.getType(), structInstant, ptrToName,
        builder.getI64ArrayAttr(0));

    builder.create<ll::ReturnOp>(loc, structInstant);
  }

  return global;
};

struct LowerSymbol : public mlir::OpConversionPattern<slir::SymbolOp> {
  using OpConversionPattern<slir::SymbolOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(serene::slir::SymbolOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

mlir::LogicalResult
LowerSymbol::matchAndRewrite(serene::slir::SymbolOp op, OpAdaptor adaptor,
                             mlir::ConversionPatternRewriter &rewriter) const {

  auto ns     = op.ns();
  auto name   = op.name();
  auto loc    = op.getLoc();
  auto module = op->getParentOfType<mlir::ModuleOp>();

  // If there is no use for the result of this op then simply erase it
  // if (!op.getResult().use_empty()) {
  //   rewriter.eraseOp(op);
  //   return mlir::success();
  // }

  auto global = getOrCreateSymbol(loc, rewriter, ns, name, module);
  rewriter.eraseOp(op);

  (void)adaptor;
  (void)global;

  return mlir::success();
}

struct LowerDefine : public mlir::OpConversionPattern<slir::DefineOp> {
  using OpConversionPattern<slir::DefineOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(serene::slir::DefineOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

mlir::LogicalResult
LowerDefine::matchAndRewrite(serene::slir::DefineOp op, OpAdaptor adaptor,
                             mlir::ConversionPatternRewriter &rewriter) const {

  (void)rewriter;
  (void)adaptor;

  auto value         = op.value();
  auto *valueop      = value.getDefiningOp();
  auto maybeTopLevel = op.is_top_level();
  bool isTopLevel    = false;

  if (maybeTopLevel) {
    isTopLevel = *maybeTopLevel;
  }

  // If the value than we bind a name to is a constant, rewrite to
  // `define_constant`

  // TODO: Replace the isConstantLike with a `hasTrait` call
  if (mlir::detail::isConstantLike(valueop)) {

    mlir::Attribute constantValue;
    if (!mlir::matchPattern(value, mlir::m_Constant(&constantValue))) {
      PASS_LOG(
          "Failure: The constant like op don't have a constant attribute.");
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<slir::DefineConstantOp>(
        op, op.sym_name(), constantValue, rewriter.getBoolAttr(isTopLevel),
        op.sym_visibilityAttr());

    // TODO: Erase the valueop if it has no other 'use' in the IR
    // rewriter.eraseOp(valueop);

    return mlir::success();
  }

  // If the value was a Function literal (like an anonymous function)
  // rewrite to a Func.FuncOp
  if (mlir::isa<slir::FnOp>(valueop)) {
    rewriter.eraseOp(op);
    return mlir::success();
  }

  // TODO: [lib] If we're building an executable `linkonce` is a good choice
  //       but for a library we need to choose a better link type
  ll::Linkage linkage = ll::Linkage::Linkonce;
  auto loc            = op.getLoc();
  auto moduleOp       = op->getParentOfType<mlir::ModuleOp>();
  auto ns             = moduleOp.getNameAttr();
  auto name           = op.getName();

  mlir::Attribute initAttr{};
  std::string fqsym;

  makeFQSymbolName(ns.getValue(), name, fqsym);

  if (!isTopLevel) {
    auto llvmType = typeConverter->convertType(value.getType());

    {
      mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
      auto moduleOp        = op->getParentOfType<mlir::ModuleOp>();
      auto &topLevelRegion = moduleOp.getBodyRegion();
      auto &moduleBlock    = topLevelRegion.getBlocks();

      rewriter.setInsertionPointToStart(&moduleBlock.front());

      auto globalOp = rewriter.create<ll::GlobalOp>(loc, llvmType,
                                                    /*isConstant=*/false,
                                                    linkage, fqsym, initAttr);
      auto &gr      = globalOp.getInitializerRegion();
      auto *block   = rewriter.createBlock(&gr);

      if (block == nullptr) {
        // TODO: use diagnastics
        llvm::errs() << "Faild to create block of the globalOp!";
        return mlir::failure();
      }

      rewriter.setInsertionPointToStart(block);

      auto undef = rewriter.create<ll::UndefOp>(loc, llvmType);
      rewriter.create<ll::ReturnOp>(loc, undef.getResult());
    }

    rewriter.setInsertionPointAfter(op);

    auto symRef = mlir::SymbolRefAttr::get(rewriter.getContext(), fqsym);
    // auto llvmValue = typeConverter->materializeTargetConversion(
    //     rewriter, loc, llvmType, value);

    // llvm::outs() << ">>> " << symRef << "|" << llvmValue << "|" << op <<
    // "\n";
    rewriter.replaceOpWithNewOp<slir::SetValueOp>(op, symRef, value);
    // auto setvalOp = rewriter.create<slir::SetValueOp>(loc, symRef,
    // llvmValue); rewriter.insert(setvalOp); rewriter.eraseOp(op);
    return mlir::success();
  }

  // auto globop = rewriter.create<ll::GlobalOp>(loc, value.getType(),
  //                                                     /*isConstant=*/false,
  //                                                     linkage, fqsym,
  //                                                     initAttr);
  // auto &gr    = globop.getInitializerRegion();
  // auto *block = rewriter.createBlock(&gr);

  // block->addArgument(value.getType(), value.getLoc());

  // rewriter.setInsertionPoint(block, block->begin());

  // rewriter.create<ll::ReturnOp>(value.getLoc(),
  // adaptor.getOperands());

  // if (!op.getResult().use_empty()) {
  //   auto symValue = rewriter.create<slir::SymbolOp>(loc, ns, name);

  //   rewriter.replaceOp(op, symValue.getResult());
  // }

  rewriter.eraseOp(op);

  return mlir::success();
}

struct LowerDefineConstant
    : public mlir::OpConversionPattern<slir::DefineConstantOp> {
  using OpConversionPattern<slir::DefineConstantOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(serene::slir::DefineConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

mlir::LogicalResult LowerDefineConstant::matchAndRewrite(
    serene::slir::DefineConstantOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  (void)rewriter;
  (void)adaptor;

  auto value    = op.value();
  auto name     = op.getName();
  auto loc      = op.getLoc();
  auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
  auto ns       = moduleOp.getNameAttr();
  std::string fqsym;

  makeFQSymbolName(ns.getValue(), name, fqsym);

  // TODO: [lib] If we're building an executable `linkonce` is a good choice
  //       but for a library we need to choose a better link type
  ll::Linkage linkage = ll::Linkage::Linkonce;

  // TODO: use ll::ConstantOp instead
  UNUSED(rewriter.create<ll::GlobalOp>(loc, value.getType(),
                                       /*isConstant=*/true, linkage, fqsym,
                                       value));

  // if (!op.value().use_empty()) {
  //   auto symValue = rewriter.create<slir::SymbolOp>(loc, ns, name);

  //   rewriter.replaceOp(op, symValue.getResult());
  // }

  rewriter.eraseOp(op);
  return mlir::success();
}

#define GEN_PASS_CLASSES
#include "serene/passes/passes.h.inc"

class LowerSLIR : public LowerSLIRBase<LowerSLIR> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    mlir::ConversionTarget target(getContext());

    slir::TypeConverter typeConverter(getContext());

    // We define the specific operations, or dialects, that are legal targets
    // for this lowering. In our case, we are lowering to the `Standard`
    // dialects.
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::arith::ArithmeticDialect>();
    target.addLegalDialect<ll::LLVMDialect>();

    // We also define the SLIR dialect as Illegal so that the conversion will
    // fail if any of these operations are *not* converted.
    target.addIllegalDialect<serene::slir::SereneDialect>();

    // Mark operations that are LEGAL for this pass. It means that we don't
    // lower them is this pass but we will in another pass. So we don't want to
    // get an error since we are not lowering them.
    // target.addLegalOp<serene::slir::PrintOp>();
    target.addLegalOp<slir::FnOp, slir::ValueOp, slir::SetValueOp>();

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the SLIR operations.
    mlir::RewritePatternSet patterns(&getContext());

    // Pattern to lower ValueOp and FnOp
    // LowerDefineConstant
    patterns.add<LowerSymbol, LowerDefine, LowerDefineConstant>(typeConverter,
                                                                &getContext());

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createLowerSLIR() {
  return std::make_unique<LowerSLIR>();
}

#define GEN_PASS_REGISTRATION
#include "serene/passes/passes.h.inc"

// ----------------------------------------------------------------------------
// ValueOp lowering to constant op
struct ValueOpLowering : public mlir::OpRewritePattern<serene::slir::Value1Op> {
  using OpRewritePattern<serene::slir::Value1Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(serene::slir::Value1Op op,
                  mlir::PatternRewriter &rewriter) const final;
};

mlir::LogicalResult
ValueOpLowering::matchAndRewrite(serene::slir::Value1Op op,
                                 mlir::PatternRewriter &rewriter) const {
  auto value         = op.value();
  mlir::Location loc = op.getLoc();

  llvm::SmallVector<mlir::Type, 1> arg_types(0);
  auto func_type = rewriter.getFunctionType(arg_types, rewriter.getI64Type());
  // TODO: use a mechanism to generate unique names
  auto fn = rewriter.create<mlir::func::FuncOp>(loc, "randomname", func_type);

  auto *entryBlock = fn.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  // Since we only support i64 at the moment we use ConstantOp
  auto retVal = rewriter
                    .create<mlir::arith::ConstantIntOp>(loc, (int64_t)value,
                                                        rewriter.getI64Type())
                    .getResult();

  UNUSED(rewriter.create<mlir::func::ReturnOp>(loc, retVal));

  fn.setPrivate();

  // Erase the original ValueOP
  rewriter.eraseOp(op);
  return mlir::success();
}

// ----------------------------------------------------------------------------
// Fn lowering pattern
struct FnOpLowering : public mlir::OpRewritePattern<serene::slir::Fn1Op> {
  using OpRewritePattern<serene::slir::Fn1Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(serene::slir::Fn1Op op,
                  mlir::PatternRewriter &rewriter) const final;
};

mlir::LogicalResult
FnOpLowering::matchAndRewrite(serene::slir::Fn1Op op,
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
  auto fn        = rewriter.create<mlir::func::FuncOp>(loc, name, func_type);

  auto *entryBlock = fn.addEntryBlock();

  rewriter.setInsertionPointToStart(entryBlock);

  auto retVal = rewriter
                    .create<mlir::arith::ConstantIntOp>(loc, (int64_t)3,
                                                        rewriter.getI64Type())
                    .getResult();

  rewriter.create<mlir::func::ReturnOp>(loc, retVal);

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
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithmeticDialect>();
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
  target.addLegalDialect<mlir::func::FuncDialect>();
  target.addLegalDialect<mlir::arith::ArithmeticDialect>();

  // We also define the SLIR dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted.
  target.addIllegalDialect<serene::slir::SereneDialect>();

  // Mark operations that are LEGAL for this pass. It means that we don't lower
  // them is this pass but we will in another pass. So we don't want to get
  // an error since we are not lowering them.
  // target.addLegalOp<serene::slir::PrintOp>();
  target.addLegalOp<mlir::func::FuncOp>();

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

void registerAllPasses() { registerPasses(); }
} // namespace serene::passes
