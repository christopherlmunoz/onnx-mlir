/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Return.cpp - ONNX Operations -------------------===//
//
// This file provides definition of ONNX dialect Return operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

namespace {
// True if rhs has the same shape as lhs or less specific shape than lhs:
// either no rank or dimensions with dynamic size where lhs has static size.
bool shapeIsSameOrMoreSpecific(ShapedType lhs, ShapedType rhs) {
  // Unranked is the least specific:
  if (!rhs.hasRank())
    return true;
  if (!lhs.hasRank())
    return false;

  // Otherwise the shapes are incompatible if ranks are different.
  if (lhs.getRank() != rhs.getRank())
    return false;

  for (auto [lhsDimSize, rhsDimSize] :
      llvm::zip(lhs.getShape(), rhs.getShape())) {
    // rhs dim size is more specific or incompatible unless it's dynamic ("?")
    // or identical to lhs dim size
    if (!(ShapedType::isDynamic(rhsDimSize) || lhsDimSize == rhsDimSize))
      return false;
  }
  return true;
}

// True if the types are the same up to shape specificity.
bool typeIsSameOrMoreSpecific(Type lhs, Type rhs) {
  ShapedType lhsShaped = mlir::dyn_cast<ShapedType>(lhs);
  ShapedType rhsShaped = mlir::dyn_cast<ShapedType>(rhs);

  if (!lhsShaped && !rhsShaped) {
    return lhs == rhs;
  }

  if (!lhsShaped || !rhsShaped) {
    // lhs and rhs must agree on whether they are ShapedType.
    return false;
  }

  if (lhsShaped.getElementType() != rhsShaped.getElementType()) {
    return false;
  }

  return shapeIsSameOrMoreSpecific(lhsShaped, rhsShaped);
}
} // namespace

// Implementation is adapted from mlir/lib/Dialect/Func/IR/FuncOps.cpp
// relaxing the type check to allow more specific shapes.
LogicalResult ONNXReturnOp::verify() {
  auto function = mlir::cast<func::FuncOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (!typeIsSameOrMoreSpecific(getOperand(i).getType(), results[i]))
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") is incompatible with function result type ("
                         << results[i] << ")"
                         << " in function @" << function.getName();

  return success();
}

//===----------------------------------------------------------------------===//
// Canonicalize
//===----------------------------------------------------------------------===//

namespace {
// Refine the enclosing function's result types to the operand types of this
// onnx.Return. Rewrites are allowed to make operand shapes more specific than
// the function signature (see the op description), but func::FuncOp
// verification requires return-like terminator operand types to match the
// signature exactly at the end of every pass, so the refinement must be
// propagated to the signature before the canonicalization driver finishes.
struct RefineFunctionTypeFromReturnPattern
    : public OpRewritePattern<ONNXReturnOp> {
  using OpRewritePattern<ONNXReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReturnOp returnOp, PatternRewriter &rewriter) const override {
    auto function = mlir::dyn_cast<func::FuncOp>(returnOp->getParentOp());
    if (!function)
      return failure();
    FunctionType funcType = function.getFunctionType();
    auto opTypes = returnOp.getOperandTypes();
    SmallVector<Type, 4> operandTypes(opTypes.begin(), opTypes.end());
    // Leave invalid or non-refining operand types for the verifier to report.
    if (operandTypes.size() != funcType.getNumResults())
      return failure();
    if (TypeRange(operandTypes) == TypeRange(funcType.getResults()))
      return failure();
    for (auto [opTy, resTy] : llvm::zip(operandTypes, funcType.getResults()))
      if (!typeIsSameOrMoreSpecific(opTy, resTy))
        return failure();
    // Only rewrite the signature when every return site agrees on the types.
    for (Block &block : function.getBody()) {
      if (block.empty())
        continue;
      if (auto other = mlir::dyn_cast<ONNXReturnOp>(&block.back()))
        if (other != returnOp &&
            TypeRange(other.getOperandTypes()) != TypeRange(operandTypes))
          return failure();
    }
    rewriter.modifyOpInPlace(function, [&] {
      function.setType(funcType.clone(funcType.getInputs(), operandTypes));
    });
    return success();
  }
};
} // namespace

void ONNXReturnOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RefineFunctionTypeFromReturnPattern>(context);
}
