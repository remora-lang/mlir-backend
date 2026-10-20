#pragma once
// CPU-dev-testing-only fallbacks.
//
// Some ops this backend emits (chlo.ragged_dot, so far) have no CPU
// ThunkEmitter implementation in XLA, even though XLA gives them
// dedicated GPU and TPU lowerings (confirmed by reading JAX's own
// `ragged_dot_general_p` lowering-rule registrations: GPU- and
// TPU-specific rules exist, with a generic dense-decomposition fallback
// used for everything else, CPU included). GPU/TPU is the real target
// for this backend; CPU only matters for exercising the compiler during
// development (no Metal/GPU plugin is installed in this environment, so
// CPU is literally the only thing locally testable).
//
// Everything in this file is therefore a deliberately separate,
// dev-testing-only accommodation: free functions, not FutharkCompiler
// methods, taking the builder/context and already-resolved operands
// explicitly rather than reaching into compiler state. Nothing here is
// reachable by default -- only behind an explicit opt-in flag
// (mlir-backend's --ragged-dot-cpu-fallback), which only run_tests.py's
// test runner sets, since test execution only ever happens on CPU here.
// A person invoking `compile`/`mlir-backend` directly still gets the
// real, performance-relevant path (LowerRaggedDot's chlo.RaggedDotOp)
// unconditionally. Never let anything in this file influence, replace,
// or live near that path.
#include "stablehlo/dialect/StablehloOps.h"
#include "utils.hpp"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include <llvm/ADT/SmallVector.h>

namespace cpu_dev_fallbacks {

namespace hlo = mlir::stablehlo;

// Inclusive cumsum of a 1-D tensor via a full-width, left-padded
// reduce_window -- the same technique LowerSegScanReduceWindow uses in
// compiler.hpp, duplicated here rather than shared across the
// class-method/free-function boundary (this file is deliberately
// self-contained, not wired into the rest of the compiler's helpers).
inline mlir::Value Cumsum(mlir::ImplicitLocOpBuilder &builder,
                          mlir::Value x) {
  auto ty = mlir::cast<mlir::RankedTensorType>(x.getType());
  mlir::Type elemTy = ty.getElementType();
  int64_t n = ty.getShape()[0];

  auto zero = hlo::ConstantOp::create(
      builder, mlir::DenseElementsAttr::get(
                   mlir::RankedTensorType::get({}, elemTy),
                   builder.getZeroAttr(elemTy)));

  auto paddingTy = mlir::RankedTensorType::get({1, 2}, builder.getI64Type());
  auto paddingAttr = mlir::DenseIntElementsAttr::get(
      paddingTy, mlir::SmallVector<int64_t>{n - 1, 0});

  auto op = hlo::ReduceWindowOp::create(
      builder, mlir::TypeRange{ty}, mlir::ValueRange{x},
      mlir::ValueRange{zero}, builder.getDenseI64ArrayAttr({n}),
      builder.getDenseI64ArrayAttr({1}), builder.getDenseI64ArrayAttr({1}),
      builder.getDenseI64ArrayAttr({1}), paddingAttr);

  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Block *blk = builder.createBlock(
      &op.getBody(), {},
      {mlir::RankedTensorType::get({}, elemTy),
       mlir::RankedTensorType::get({}, elemTy)},
      {builder.getLoc(), builder.getLoc()});
  builder.setInsertionPointToStart(blk);
  auto sum =
      hlo::AddOp::create(builder, blk->getArgument(0), blk->getArgument(1));
  hlo::ReturnOp::create(builder, mlir::ValueRange{sum});

  return op->getResult(0);
}

// Dense-masked-matmul decomposition of ragged_dot, mirrored directly from
// jax.lax.ragged_dot's own generic (non-GPU/TPU) lowering rule -- checked
// via `jax.jit(f).lower(...).as_text()` for this exact shape, not
// re-derived from the algorithm description alone. Scoped to the one
// shape this is actually needed for: "mode 1" with no batch dimensions
// (lhs: [m,k], rhs: [g,k,n], group_sizes: [g] -> [m,n], ragged dimension
// on lhs, a single contracting dimension) -- the shape
// tests/static_shapes/ragged_dot_mode1.fut and moe_black_box.fut's
// ragged_matmul both use. Undefined() on anything else (batch dimensions,
// mode 2's ragged-contracting-dimension shape, mode 3's ragged-batch-
// dimension shape) rather than replicating JAX's full generality for
// cases nothing here needs -- mode 3 doesn't need this fallback at all
// (chlo.RaggedDotOp already executes on XLA:CPU for it, since XLA
// algebraically simplifies away a ragged batch dimension's group_sizes
// dependence).
// Whether LowerRaggedDotCpuFallback below actually supports this shape
// (mode 1, no batch dims, single contracting dim -- see that function's
// comment). Callers check this first and keep the real chlo.RaggedDotOp
// path for every shape the fallback can't handle, rather than dispatching
// unconditionally whenever the --ragged-dot-cpu-fallback flag is set --
// that matters for shapes like mode 3's (ragged dim == batch dim) that
// already execute fine on XLA:CPU via chlo.RaggedDotOp directly, so there's
// no reason to route them through this narrower, CPU-only decomposition
// at all.
inline bool IsRaggedDotCpuFallbackSupported(
    mlir::Value lhs, mlir::Value rhs,
    llvm::ArrayRef<int64_t> lhsBatchingDimensions,
    llvm::ArrayRef<int64_t> lhsContractingDimensions, int64_t lhsRaggedDim,
    llvm::ArrayRef<int64_t> rhsGroupDimensions) {
  auto lhsTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsTy = mlir::cast<mlir::RankedTensorType>(rhs.getType());
  return lhsBatchingDimensions.empty() && lhsTy.getRank() == 2 &&
         rhsTy.getRank() == 3 && lhsContractingDimensions.size() == 1 &&
         lhsContractingDimensions[0] == 1 && lhsRaggedDim == 0 &&
         rhsGroupDimensions.size() == 1 && rhsGroupDimensions[0] == 0;
}

inline mlir::Value LowerRaggedDotCpuFallback(
    mlir::ImplicitLocOpBuilder &builder, mlir::MLIRContext &context,
    mlir::Value lhs, mlir::Value rhs, mlir::Value groupSizes,
    llvm::ArrayRef<int64_t> lhsBatchingDimensions,
    llvm::ArrayRef<int64_t> lhsContractingDimensions, int64_t lhsRaggedDim,
    llvm::ArrayRef<int64_t> rhsGroupDimensions,
    llvm::ArrayRef<int64_t> resultShape, mlir::ArrayAttr precisionConfig) {
  auto lhsTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsTy = mlir::cast<mlir::RankedTensorType>(rhs.getType());
  mlir::Type elemTy = lhsTy.getElementType();

  if (!lhsBatchingDimensions.empty() || lhsTy.getRank() != 2 ||
      rhsTy.getRank() != 3 || lhsContractingDimensions.size() != 1 ||
      lhsContractingDimensions[0] != 1 || lhsRaggedDim != 0 ||
      rhsGroupDimensions.size() != 1 || rhsGroupDimensions[0] != 0)
    Undefined("ragged_dot CPU fallback: only mode 1 with no batch "
              "dimensions and a single contracting dimension is "
              "supported");

  int64_t m = lhsTy.getShape()[0];
  int64_t k = lhsTy.getShape()[1];
  int64_t g = rhsTy.getShape()[0];
  int64_t n = rhsTy.getShape()[2];
  (void)k;

  // Exclusive prefix sum of group_sizes (per-group start offsets):
  // prepend 0 to the inclusive cumsum and drop its last element.
  auto groupSizesTy = mlir::cast<mlir::RankedTensorType>(groupSizes.getType());
  mlir::Type gsElemTy = groupSizesTy.getElementType();
  mlir::Value inclusive = Cumsum(builder, groupSizes);
  mlir::Value zeroScalar = hlo::ConstantOp::create(
      builder, mlir::DenseElementsAttr::get(
                   mlir::RankedTensorType::get({}, gsElemTy),
                   builder.getZeroAttr(gsElemTy)));
  mlir::Value zeroVec = hlo::BroadcastInDimOp::create(
      builder, mlir::RankedTensorType::get({g}, gsElemTy), zeroScalar,
      mlir::SmallVector<int64_t>{});
  mlir::Value zeroHead = hlo::SliceOp::create(
      builder, zeroVec, mlir::SmallVector<int64_t>{0},
      mlir::SmallVector<int64_t>{1}, mlir::SmallVector<int64_t>{1});
  mlir::Value cumsumHead = hlo::SliceOp::create(
      builder, inclusive, mlir::SmallVector<int64_t>{0},
      mlir::SmallVector<int64_t>{g - 1}, mlir::SmallVector<int64_t>{1});
  mlir::Value starts = hlo::ConcatenateOp::create(
      builder, mlir::ValueRange{zeroHead, cumsumHead}, (uint64_t)0);

  // Broadcast lhs to [g, m, k], build the per-group row mask (start <=
  // row_index < end, row_index = iota over the m dimension), and select
  // the masked lhs against zero.
  auto gmkTy = mlir::RankedTensorType::get({g, m, k}, elemTy);
  mlir::Value lhsB = hlo::BroadcastInDimOp::create(
      builder, gmkTy, lhs, mlir::SmallVector<int64_t>{1, 2});
  auto gmkIdxTy = mlir::RankedTensorType::get({g, m, k}, gsElemTy);
  mlir::Value rowIdx = hlo::IotaOp::create(builder, gmkIdxTy,
                                           builder.getI64IntegerAttr(1));
  mlir::Value endsB = hlo::BroadcastInDimOp::create(
      builder, gmkIdxTy, inclusive, mlir::SmallVector<int64_t>{0});
  mlir::Value startsB = hlo::BroadcastInDimOp::create(
      builder, gmkIdxTy, starts, mlir::SmallVector<int64_t>{0});
  mlir::Value ge = hlo::CompareOp::create(builder, startsB, rowIdx,
                                          hlo::ComparisonDirection::LE,
                                          hlo::ComparisonType::SIGNED);
  mlir::Value lt = hlo::CompareOp::create(builder, rowIdx, endsB,
                                          hlo::ComparisonDirection::LT,
                                          hlo::ComparisonType::SIGNED);
  mlir::Value mask = hlo::AndOp::create(builder, ge, lt);
  mlir::Value zeroElem = hlo::ConstantOp::create(
      builder, mlir::DenseElementsAttr::get(
                   mlir::RankedTensorType::get({}, elemTy),
                   builder.getZeroAttr(elemTy)));
  mlir::Value zeroGmk = hlo::BroadcastInDimOp::create(
      builder, gmkTy, zeroElem, mlir::SmallVector<int64_t>{});
  mlir::Value masked = hlo::SelectOp::create(builder, mask, lhsB, zeroGmk);

  // One dot_general contracting [g, k] jointly against rhs's [g, k] --
  // since each row's mask is nonzero for exactly one group, this sums
  // only that group's real contribution and zeroes out every other
  // group's, with no reshape/concatenate needed.
  auto dotDims = hlo::DotDimensionNumbersAttr::get(
      &context, /*lhsBatchingDimensions=*/{}, /*rhsBatchingDimensions=*/{},
      /*lhsContractingDimensions=*/{0, 2}, /*rhsContractingDimensions=*/{0, 1});
  auto resultTy = mlir::RankedTensorType::get(resultShape, elemTy);
  auto dot = hlo::DotGeneralOp::create(builder, resultTy, masked, rhs,
                                       dotDims, precisionConfig, {});
  (void)n;
  return dot->getResult(0);
}

} // namespace cpu_dev_fallbacks
