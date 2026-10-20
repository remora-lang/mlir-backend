#pragma once
// Dummy lowering for Futhark's "builtin/repiota" library function.
//
// repiota(g, ns) computes, conceptually, np.repeat(arange(g), ns): for each
// group i in 0..g, repeat the group index i exactly ns[i] times,
// concatenated -- a segment-id array over a run-length-expanded output
// whose total length (the existential `r` in its own declared return
// type, `?[r].(i64, *[r]bool, *[g]i64, *[r]i64)`) is data-dependent (it
// depends on ns's actual runtime values, not just its shape). That
// data-dependent size has no valid representation in this backend's
// StableHLO target (every shape in the module must be static -- see
// compiler.hpp's LowerFunction), so repiota's *real* semantics can never
// be lowered here at all, used or not.
//
// "builtin/repiota" is a name the Futhark compiler itself generates (never
// user-written), so it's safe to intercept it unconditionally by exact
// name and hand back a cheap, fixed-shape placeholder instead of its real
// (unlowerable) body -- see compiler.hpp's ExpApply, the only caller. This
// is only correct because whatever recognized idiom consumes repiota's
// result (see ragged_unflatten_gather.hpp) replaces that consumer entirely
// rather than ever reading the placeholder's actual values; if a future
// test needs repiota's real result for something that *isn't* replaced
// this way, this dummy would silently produce a wrong answer instead of
// failing loudly -- which is why every op this file builds is tagged with
// the kDummyAttr attribute (shared with ragged_unflatten_gather.hpp's own,
// separately-defined, same-value marker). mlir-backend.cpp asserts none
// survive the post-lowering DCE pass before the module is considered
// final.
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace builtin_repiota {

namespace hlo = mlir::stablehlo;

inline constexpr char kDummyAttr[] = "futhark.dummy";

// `g` is the (already-known-static) number of groups, from the call's own
// first argument. Returns repiota's 4 outputs in order: m (existential
// total length, here just 0), flags ([0]bool), offsets ([g]i64), elems
// ([0]i64). Only offsets' shape matches repiota's real declared type
// ([g]i64, not existential) -- the other three are existentially sized in
// the real signature, so any fixed placeholder shape is equally
// "wrong" there, and 0 is the cheapest.
inline llvm::SmallVector<mlir::Value, 4>
LowerDummy(mlir::ImplicitLocOpBuilder &builder, int64_t g) {
  auto i64Ty = builder.getI64Type();
  auto boolTy = builder.getI1Type();

  auto zeroConst = [&](llvm::ArrayRef<int64_t> shape, mlir::Type elemTy) {
    auto ty = mlir::RankedTensorType::get(shape, elemTy);
    auto op = hlo::ConstantOp::create(
        builder, mlir::DenseElementsAttr::get(ty, builder.getZeroAttr(elemTy)));
    op->setAttr(kDummyAttr, builder.getUnitAttr());
    return op->getResult(0);
  };

  mlir::Value m = zeroConst({}, i64Ty);
  mlir::Value flags = zeroConst({0}, boolTy);
  mlir::Value offsets = zeroConst({g}, i64Ty);
  mlir::Value elems = zeroConst({0}, i64Ty);
  return {m, flags, offsets, elems};
}

} // namespace builtin_repiota
