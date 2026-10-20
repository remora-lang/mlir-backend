#pragma once
// Exclusive-scan source-idiom recognizer.
//
// Futhark's own canonical lowering of an EXCLUSIVE scan (as opposed to the
// INCLUSIVE one this backend already supports via LowerSegScanReduceWindow)
// expresses it as an ordinary SegScan whose per-element ("mapped") kernel
// body is `gtid == 0 ? <literal> : arr[gtid - 1]` -- a plain array shifted
// right by one, not a real per-element computation:
//
//   let cond = gtid == 0
//   let to_prescan = if cond then <literal> else { let i = gtid - 1
//                                                   let x = arr[i]
//                                                   in x }
//   in to_prescan
//
// compiler.hpp's LowerSegMap can't route this through the general affine/
// gather-read machinery at all: the read is both offset (`gtid - 1`, not a
// bare iteration id) and nested inside an ExpIf branch rather than a
// top-level kernel-body statement. First hit getting the custom-flattening
// feature's irregular-layout offset computation (an exclusive prefix sum
// over segment sizes) working -- no earlier test needed an exclusive scan.
//
// This file only recognizes the shape -- pure inspection of the Futhark IR
// (KernelBody/Stm), no MLIR involved, so it can run before any lowering is
// attempted. compiler.hpp's LowerSegScan does the actual emission, reusing
// its own already-verified LowerSegScanReduceWindow directly on the
// shifted array (see its call site for why that's equivalent: an exclusive
// scan of `arr` is exactly an inclusive scan of `arr` shifted right by one
// with the literal filled in at position 0).
#include "ir/segop.hpp"
#include <optional>

namespace exclusive_scan {

struct Match {
  VName array;
  SubExp fillValue;
};

// `iterId` is the segscan's sole iteration variable (callers only try this
// for rank-1 segscans; a segmented/multi-dim exclusive scan isn't a shape
// this idiom covers and isn't needed yet).
inline std::optional<Match> TryMatch(const std::string &iterId,
                                     const KernelBody &body) {
  if (body.stms.size() != 2 || body.result.size() != 1)
    return std::nullopt;

  auto &condStm = body.stms[0];
  auto *condExp = std::get_if<ExpBasicOp>(&condStm.exp.v);
  auto *cmp = condExp ? std::get_if<BasicOpCmpOp>(&condExp->op.v) : nullptr;
  if (condStm.pat.elems.size() != 1 || !cmp)
    return std::nullopt;
  auto isIterZero = [&](const SubExp &a, const SubExp &b) {
    auto *var = std::get_if<VarSubExp>(&a.v);
    auto *lit = std::get_if<ConstantSubExp>(&b.v);
    return var && var->v.name == iterId && lit && lit->GetIntValue() == 0;
  };
  if (!isIterZero(cmp->op0, cmp->op1) && !isIterZero(cmp->op1, cmp->op0))
    return std::nullopt;
  const std::string &condName = condStm.pat.elems[0].name.name;

  auto &scanStm = body.stms[1];
  auto *ifExp = std::get_if<ExpIf>(&scanStm.exp.v);
  if (scanStm.pat.elems.size() != 1 || !ifExp)
    return std::nullopt;
  auto *condVar = std::get_if<VarSubExp>(&ifExp->cond.v);
  if (!condVar || condVar->v.name != condName)
    return std::nullopt;
  if (!ifExp->then_body->stms.empty() ||
      ifExp->then_body->result.size() != 1)
    return std::nullopt;
  SubExp thenVal = ifExp->then_body->result[0];

  auto &elseBody = *ifExp->else_body;
  if (elseBody.stms.size() != 2 || elseBody.result.size() != 1)
    return std::nullopt;
  auto &shiftStm = elseBody.stms[0];
  auto &readStm = elseBody.stms[1];
  if (shiftStm.pat.elems.size() != 1 || readStm.pat.elems.size() != 1)
    return std::nullopt;

  auto *shiftExp = std::get_if<ExpBasicOp>(&shiftStm.exp.v);
  auto *sub = shiftExp ? std::get_if<BasicOpBinOp>(&shiftExp->op.v) : nullptr;
  if (!sub)
    return std::nullopt;
  auto *shiftVar = std::get_if<VarSubExp>(&sub->op0.v);
  auto *shiftLit = std::get_if<ConstantSubExp>(&sub->op1.v);
  if (!shiftVar || shiftVar->v.name != iterId || !shiftLit ||
      shiftLit->GetIntValue() != 1)
    return std::nullopt;

  auto *readExp = std::get_if<ExpBasicOp>(&readStm.exp.v);
  auto *idx = readExp ? std::get_if<BasicOpIndex>(&readExp->op.v) : nullptr;
  if (!idx || idx->slice.dims.size() != 1)
    return std::nullopt;
  auto *fix = std::get_if<DimFix<SubExp>>(&idx->slice.dims[0].v);
  auto *idxVar = fix ? std::get_if<VarSubExp>(&fix->i.v) : nullptr;
  if (!idxVar || idxVar->v.name != shiftStm.pat.elems[0].name.name)
    return std::nullopt;

  auto *elseResVar = std::get_if<VarSubExp>(&elseBody.result[0].v);
  if (!elseResVar || elseResVar->v.name != readStm.pat.elems[0].name.name)
    return std::nullopt;

  auto *bodyResVar = std::get_if<VarSubExp>(&body.result[0].result.v);
  if (!bodyResVar || bodyResVar->v.name != scanStm.pat.elems[0].name.name)
    return std::nullopt;

  return Match{idx->vName, thenVal};
}

} // namespace exclusive_scan
