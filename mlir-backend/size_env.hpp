#pragma once
// Backfills a call's sizeEnv (see compiler.hpp's ExpApply and
// LowerFunction) from array-typed arguments, for a symbolic size that
// can't be resolved from its own (non-literal) scalar argument.
//
// sizeEnv's original mechanism only resolves a callee's named size
// (e.g. `a_k` in `a_data: [a_k]f32`) from a *literal* scalar argument at
// the call site. That's not enough on its own for a custom-flattening
// flatten target like `matmul_nonuniform`: its own `a_k` argument is a
// genuinely dynamic value in the Futhark IR (the flattened buffer's
// length, computed from runtime data), so there's no literal to read.
// But by the time this call is reached, `a_data`'s own argument (the
// flattened buffer itself) may already be bound to an actual MLIR value
// whose type is already static -- e.g. because the gather that used to
// produce it was recognized and replaced by a plain reshape (see
// ragged_unflatten_gather.hpp). When that's true, `a_k` is recoverable
// from `a_data`'s own already-resolved shape instead of from `a_k`'s own
// unresolved argument.
//
// This file only does the structural matching (which array param's
// shape mentions which still-unresolved symbolic name, and which
// position); the one thing it can't do standalone is ask "is this
// argument name already bound, and to what MLIR type" (Ctx is
// compiler.hpp-internal), so that's injected via `lookupType` rather
// than making this file depend on Ctx at all.
#include "ir/segop.hpp"
#include "mlir/IR/BuiltinTypes.h"
#include <functional>
#include <optional>
#include <unordered_map>

namespace size_env {

inline void BackfillFromArrayArgs(
    const std::vector<FParam> &params,
    const std::vector<std::pair<SubExp, Diet>> &args,
    const std::function<std::optional<mlir::Type>(const std::string &)>
        &lookupType,
    std::unordered_map<std::string, int64_t> &sizeEnv) {
  for (auto [param, argDiet] : llvm::zip_equal(params, args)) {
    auto *arrTy =
        std::get_if<TypeArray<Shape, NoUniqueness>>(&param.dec.v.t.v);
    if (!arrTy)
      continue;
    auto *argVar = std::get_if<VarSubExp>(&argDiet.first.v);
    if (!argVar)
      continue;
    auto ty = lookupType(argVar->v.name);
    if (!ty)
      continue;
    auto rankedTy = mlir::dyn_cast<mlir::RankedTensorType>(*ty);
    if (!rankedTy || !rankedTy.hasStaticShape())
      continue;
    if ((int64_t)arrTy->shape.dims.size() != rankedTy.getRank())
      continue; // declared rank disagrees with the actual value -- don't
                // guess which dims line up.
    for (auto [i, dim] : llvm::enumerate(arrTy->shape.dims)) {
      auto *dimVar = std::get_if<VarSubExp>(&dim.v);
      if (!dimVar || sizeEnv.contains(dimVar->v.name))
        continue; // not a named size, or already resolved -- don't
                  // overwrite an existing (e.g. literal-derived) entry.
      sizeEnv[dimVar->v.name] = rankedTy.getShape()[i];
    }
  }
}

} // namespace size_env
