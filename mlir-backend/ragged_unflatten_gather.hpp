#pragma once
// Recognizer + dummy helper for the custom-flattening feature's ragged-
// layout "unflatten" idiom.
//
// A ragged matmul's flatten target packs each group's rows of a 2-D lhs
// array into one flat buffer. Futhark's own flattening pass expresses the
// per-flat-position read as a map computing, from the flat index `gtid`:
//   segment        = <some segment-id array>[gtid]        -- which group
//   segment_start  = <flat start offsets>[segment]
//   o              = <the caller's own row offsets>[segment]
//   row            = o + (gtid - segment_start) / k
//   col            = (gtid - segment_start) - k * ((gtid - segment_start) / k)
//   result         = lhs[row, col]
//
// Under the standard ragged_dot mode-1 convention (the caller's row
// offsets are exactly the exclusive prefix sum of the group sizes, so
// groups tile lhs's rows contiguously, in order, starting at 0), this
// always simplifies to `row = gtid / k, col = gtid % k` -- i.e. the whole
// computation is provably just `reshape(lhs, [m*k])`, row-major, nothing
// gathered at all. (Checked by hand against the actual GPU IR, not
// derived from the idiom's description alone -- see PROGRESS.md.)
//
// TryMatchGather below recognizes this *exact* arithmetic shape -- no
// general scalar-expression resolution, no tolerance for a differently-
// decomposed div/mod, no lookahead into surrounding statements. Any
// deviation bails (returns nullopt) and the caller falls through to the
// normal (currently failing, for this shape) SegMap lowering. This is a
// narrow, rigid, single-purpose optimization in the same spirit as
// exclusive_scan.hpp's shift-pattern matcher -- not a generalization of
// this backend's gather support.
//
// The *other* half of this idiom -- an initial `replicate` of a
// genuinely dynamic shape, feeding a with_acc that builds a parallel
// "which flat positions are real" flags array -- has no valid StableHLO
// representation here at all (its length is data-dependent, same root
// cause as builtin_repiota.hpp's problem), regardless of whether this
// gather match fires. See LowerDummyReplicate: same unconditional-by-
// shape, DCE-plus-safety-net treatment as builtin/repiota's dummy, for
// the same reason (compiler-generated bookkeeping with no valid static
// lowering, dead or not, once the gather above is replaced).
#include "ir/segop.hpp"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "stablehlo/dialect/StablehloOps.h"
#include <optional>

namespace ragged_unflatten {

namespace hlo = mlir::stablehlo;

// Shared with builtin_repiota.hpp's own (separately defined, same value)
// marker -- both files tag every op they build so mlir-backend.cpp's
// post-DCE pass can assert none survive into the final module.
inline constexpr char kDummyAttr[] = "futhark.dummy";

struct GatherMatch {
  VName array; // the 2-D array actually being read (e.g. lhs)
  int64_t k;   // the matched divisor/modulus -- caller should check this
               // equals `array`'s own second dimension before trusting
               // the match (see compiler.hpp's call site).
};

inline std::optional<GatherMatch> TryMatchGather(const std::string &iterId,
                                                 const KernelBody &body) {
  if (body.stms.size() != 9 || body.result.size() != 1)
    return std::nullopt;

  auto boundName = [](const Stm &stm) -> const std::string * {
    if (stm.pat.elems.size() != 1)
      return nullptr;
    return &stm.pat.elems[0].name.name;
  };
  auto asIndex = [](const Stm &stm) -> const BasicOpIndex * {
    auto *e = std::get_if<ExpBasicOp>(&stm.exp.v);
    return e ? std::get_if<BasicOpIndex>(&e->op.v) : nullptr;
  };
  auto asBinOp = [](const Stm &stm) -> const BasicOpBinOp * {
    auto *e = std::get_if<ExpBasicOp>(&stm.exp.v);
    return e ? std::get_if<BasicOpBinOp>(&e->op.v) : nullptr;
  };
  auto dimVar = [](const DimIndex<SubExp> &dim) -> const VarSubExp * {
    auto *fix = std::get_if<DimFix<SubExp>>(&dim.v);
    return fix ? std::get_if<VarSubExp>(&fix->i.v) : nullptr;
  };
  auto subVar = [](const SubExp &s) { return std::get_if<VarSubExp>(&s.v); };
  auto subLit = [](const SubExp &s) {
    return std::get_if<ConstantSubExp>(&s.v);
  };

  // stm0: segment = S[gtid]
  auto *n0 = boundName(body.stms[0]);
  auto *idx0 = asIndex(body.stms[0]);
  if (!n0 || !idx0 || idx0->slice.dims.size() != 1)
    return std::nullopt;
  auto *idx0v = dimVar(idx0->slice.dims[0]);
  if (!idx0v || idx0v->v.name != iterId)
    return std::nullopt;

  // stm1: segment_start = S2[segment]
  auto *n1 = boundName(body.stms[1]);
  auto *idx1 = asIndex(body.stms[1]);
  if (!n1 || !idx1 || idx1->slice.dims.size() != 1)
    return std::nullopt;
  auto *idx1v = dimVar(idx1->slice.dims[0]);
  if (!idx1v || idx1v->v.name != *n0)
    return std::nullopt;

  // stm2: o = S3[segment]
  auto *n2 = boundName(body.stms[2]);
  auto *idx2 = asIndex(body.stms[2]);
  if (!n2 || !idx2 || idx2->slice.dims.size() != 1)
    return std::nullopt;
  auto *idx2v = dimVar(idx2->slice.dims[0]);
  if (!idx2v || idx2v->v.name != *n0)
    return std::nullopt;

  // stm3: binop_x = gtid - segment_start
  auto *n3 = boundName(body.stms[3]);
  auto *bo3 = asBinOp(body.stms[3]);
  if (!n3 || !bo3 || !std::holds_alternative<BinOpSub>(bo3->op.v))
    return std::nullopt;
  auto *bo3a = subVar(bo3->op0);
  auto *bo3b = subVar(bo3->op1);
  if (!bo3a || bo3a->v.name != iterId || !bo3b || bo3b->v.name != *n1)
    return std::nullopt;

  // stm4: binop_y = binop_x / k  (signed quotient)
  auto *n4 = boundName(body.stms[4]);
  auto *bo4 = asBinOp(body.stms[4]);
  if (!n4 || !bo4 || !std::holds_alternative<BinOpSQuot>(bo4->op.v))
    return std::nullopt;
  auto *bo4a = subVar(bo4->op0);
  auto *bo4k = subLit(bo4->op1);
  if (!bo4a || bo4a->v.name != *n3 || !bo4k)
    return std::nullopt;
  int64_t k = bo4k->GetIntValue();

  // stm5: i_row = o + binop_y
  auto *n5 = boundName(body.stms[5]);
  auto *bo5 = asBinOp(body.stms[5]);
  if (!n5 || !bo5 || !std::holds_alternative<BinOpAdd>(bo5->op.v))
    return std::nullopt;
  auto *bo5a = subVar(bo5->op0);
  auto *bo5b = subVar(bo5->op1);
  if (!bo5a || bo5a->v.name != *n2 || !bo5b || bo5b->v.name != *n4)
    return std::nullopt;

  // stm6: binop_y2 = k * binop_y
  auto *n6 = boundName(body.stms[6]);
  auto *bo6 = asBinOp(body.stms[6]);
  if (!n6 || !bo6 || !std::holds_alternative<BinOpMul>(bo6->op.v))
    return std::nullopt;
  auto *bo6k = subLit(bo6->op0);
  auto *bo6b = subVar(bo6->op1);
  if (!bo6k || bo6k->GetIntValue() != k || !bo6b || bo6b->v.name != *n4)
    return std::nullopt;

  // stm7: i_col = binop_x - binop_y2
  auto *n7 = boundName(body.stms[7]);
  auto *bo7 = asBinOp(body.stms[7]);
  if (!n7 || !bo7 || !std::holds_alternative<BinOpSub>(bo7->op.v))
    return std::nullopt;
  auto *bo7a = subVar(bo7->op0);
  auto *bo7b = subVar(bo7->op1);
  if (!bo7a || bo7a->v.name != *n3 || !bo7b || bo7b->v.name != *n6)
    return std::nullopt;

  // stm8: v = L[i_row, i_col]
  auto *n8 = boundName(body.stms[8]);
  auto *idx8 = asIndex(body.stms[8]);
  if (!n8 || !idx8 || idx8->slice.dims.size() != 2)
    return std::nullopt;
  auto *idx8row = dimVar(idx8->slice.dims[0]);
  auto *idx8col = dimVar(idx8->slice.dims[1]);
  if (!idx8row || idx8row->v.name != *n5 || !idx8col ||
      idx8col->v.name != *n7)
    return std::nullopt;

  auto *resVar = subVar(body.result[0].result);
  if (!resVar || resVar->v.name != *n8)
    return std::nullopt;

  return GatherMatch{idx8->vName, k};
}

// See this file's header comment. The exact placeholder size never
// matters (1 is as good as any), since nothing in this idiom's design
// relies on the result faithfully matching the real, data-dependent
// length -- it's always paired with a with_acc this backend's existing
// LowerWithAcc already handles correctly once given *any* validly-typed
// `dest`.
inline mlir::Value LowerDummyReplicate(mlir::ImplicitLocOpBuilder &builder,
                                       mlir::Type elemTy) {
  auto ty = mlir::RankedTensorType::get({1}, elemTy);
  auto op = hlo::ConstantOp::create(
      builder, mlir::DenseElementsAttr::get(ty, builder.getZeroAttr(elemTy)));
  op->setAttr(kDummyAttr, builder.getUnitAttr());
  return op->getResult(0);
}

} // namespace ragged_unflatten
