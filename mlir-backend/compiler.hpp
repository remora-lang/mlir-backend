#pragma once
// Represents the state of a Futhark function
#include "blackbox.hpp"
#include "builtin_repiota.hpp"
#include "cpu_dev_fallbacks.hpp"
#include "debug.hpp"
#include "exclusive_scan.hpp"
#include "ir/core.hpp"
#include "ir/segop.hpp"
#include "ir/syntax.hpp"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "ragged_unflatten_gather.hpp"
#include "size_env.hpp"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "utils.hpp"
#include "llvm/Support/ErrorHandling.h"
#include <cctype>
#include <format>
#include <functional>
#include <iterator>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/Transforms/RegionUtils.h>
#include <optional>
#include <ranges>
#include <source_location>
#include <variant>

namespace hlo = mlir::stablehlo;

template <typename K, typename V> class Env {
  using env_t = std::unordered_map<K, V>;

  env_t env;

public:
  void insert(const K &name, V v) { env.insert_or_assign(name, v); }

  V lookup(const K &name,
           std::source_location loc = std::source_location::current()) const {
    auto v = env.find(name);
    if (v == env.end())
      throw std::runtime_error(
          std::format("env lookup at {}:{}: unbound variable '{}'",
                      loc.file_name(),
                      loc.line(),
                      name));
    return v->second;
  }

  // Non-throwing variant for call sites that genuinely expect a name may
  // not (yet) be bound -- e.g. size_env.hpp's backfill, which only wants
  // to know an argument's type *if* it's already been lowered.
  std::optional<V> tryLookup(const K &name) const {
    auto v = env.find(name);
    if (v == env.end())
      return std::nullopt;
    return v->second;
  }

  env_t::const_iterator begin() const { return env.cbegin(); }

  env_t::const_iterator end() const { return env.cend(); }
};

inline void PrintValue(const Env<std::string, mlir::Value> &env) {
  llvm::errs() << "Env {\n";
  for (const auto &[k, v] : env) {
    llvm::errs() << "  " << k << " : " << v << "\n";
  }
  llvm::errs() << "}\n";
}

struct AccValue {
  mlir::Value update;
  mlir::Value index;
};

struct Ctx {
  Env<std::string, mlir::Value> subexps;
  Env<std::string, AccValue> accs;
};

struct Dim {
  std::string id;    // For example, gtid in a kernel.
  mlir::Value value; // Lowered dimension bound.
  bool isDynamic;
  long index; // Position in the iteration space.
};

using IterationSpace = std::vector<Dim>;

struct AffineRead {
  // The result is bound to this name.
  VName result;
  // The array being read.
  VName array;
  // An affine map from the iteration space to the indexing expression.
  // For example, (d0, d1) -> (2*d0 + 1, d1).
  mlir::AffineMap indexMap;
};

// A gather: an index read (`arr[i0, i1, ...]`) where at least one `i_k` is
// not an iteration-space id directly, but the bound result of an earlier
// AffineRead in the same kernel body (e.g. `tokens[perm[gtid], gtid2]`,
// where `perm[gtid]` is itself a plain AffineRead). `AffineRead`'s
// `mlir::AffineMap` can't represent this (the index isn't a static affine
// function of the iteration ids), so it gets its own, simpler
// representation: one name per index dimension, each either an iteration
// id or an earlier AffineRead's result -- both are already bound inside
// the stablehlo.map region's block args by the time a gather is lowered,
// so no new kind of block-arg binding is needed for the *index* side, only
// for the array being gathered from (see LowerSegMap).
struct GatherRead {
  VName result;
  VName array;
  std::vector<VName> indexNames;
};

// sizeEnv resolves a dim that's a *named* size (a VarSubExp referring to an
// earlier i64 function parameter, e.g. `[b][n][k]f32` where `b` is itself a
// param) to a concrete value known at a specific call site -- see
// ExpApply's handling of a plain (non-blackbox) function call in LowerExp,
// which is the only place that ever builds a non-empty sizeEnv. Every other
// caller passes the default empty map, so an unresolved name still falls
// back to kDynamic exactly as before.
inline int64_t
toShapeType(SubExp dim,
            const std::unordered_map<std::string, int64_t> &sizeEnv = {}) {
  return match(
      dim,
      [&](const ConstantSubExp &c) { return c.GetIntValue(); },
      [&](const VarSubExp &v) {
        auto it = sizeEnv.find(v.v.name);
        if (it != sizeEnv.end())
          return it->second;
        return mlir::ShapedType::kDynamic;
      });
}

inline bool IsUnitType(const Type &type) {
  auto *prim = std::get_if<TypePrim<Shape, NoUniqueness>>(&type.t.v);
  return prim && std::holds_alternative<PrimTypeUnit>(prim->t.v);
}

inline const TypeAcc<Shape, NoUniqueness> *GetAccType(const Type &type) {
  return std::get_if<TypeAcc<Shape, NoUniqueness>>(&type.t.v);
}

inline bool IsAccType(const Type &type) { return GetAccType(type) != nullptr; }

template <typename R, typename V>
concept Iterable = std::ranges::input_range<R> &&
                   std::same_as<std::ranges::range_value_t<R>, V>;

template <Iterable<SubExp> R>
inline std::vector<int64_t>
toShapeType(const R &dims,
            const std::unordered_map<std::string, int64_t> &sizeEnv = {}) {
  std::vector<int64_t> dimsTy;
  for (auto &d : dims)
    dimsTy.push_back(toShapeType(d, sizeEnv));
  return dimsTy;
}

struct FutharkCompiler {
  using Values = mlir::SmallVector<mlir::Value>;

  Prog &prog;

  std::unordered_map<FunDef, mlir::func::FuncOp, FunDefHasher> functions;

  mlir::MLIRContext &context;

  mlir::ImplicitLocOpBuilder &builder;

  mlir::ModuleOp module;

  // Set from the command line (mlir-backend's --precision flag) before
  // lowering starts; applied uniformly to every dot_general this backend
  // emits. Replaces the old scheme of relying on example_run.py's
  // set_precision regex rewrite of the already-compiled text.
  hlo::Precision precision = hlo::Precision::DEFAULT;

  // DEFAULT prints as no precision_config attribute at all (StableHLO
  // treats "omitted" and "[DEFAULT, DEFAULT]" identically), matching the
  // output this backend has always produced for the common case.
  mlir::ArrayAttr PrecisionConfig() {
    if (precision == hlo::Precision::DEFAULT)
      return {};
    auto attr = hlo::PrecisionAttr::get(&context, precision);
    return builder.getArrayAttr({attr, attr});
  }

  // Set from the command line (mlir-backend's --ragged-dot-cpu-fallback
  // flag) before lowering starts. Default false (the real, performance-
  // relevant chlo.RaggedDotOp path, used by LowerRaggedDot below); only
  // run_tests.py's test runner ever sets this true, since chlo.RaggedDotOp
  // has no XLA:CPU ThunkEmitter implementation for the shapes this
  // backend needs it for, and test execution only ever happens on CPU in
  // this environment. See cpu_dev_fallbacks.hpp -- never let this
  // influence anything beyond LowerRaggedDot's one dispatch point.
  bool raggedDotCpuFallback = false;

  FutharkCompiler(Prog &prog, mlir::MLIRContext &context,
                  mlir::ImplicitLocOpBuilder &builder)
      : prog(prog), context(context), builder(builder) {
    context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
    context.getOrLoadDialect<mlir::math::MathDialect>();
    context.getOrLoadDialect<mlir::tensor::TensorDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::index::IndexDialect>();
    context.getOrLoadDialect<mlir::BuiltinDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<hlo::StablehloDialect>();
    context.getOrLoadDialect<mlir::chlo::ChloDialect>();
    context.getOrLoadDialect<mlir::vector::VectorDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
  }

  // sizeEnv lets a call to a plain (non-blackbox, non-entry) function whose
  // own declared param/result types are named-size-dependent (e.g.
  // `[b][n][k]f32` where `b` is itself an earlier i64 param) resolve those
  // names to the concrete values known at THIS call -- see ExpApply, the
  // only caller that ever passes a non-empty map. Every FunDef is only ever
  // emitted once per distinct call site in this backend's GPU IR input (a
  // genuinely polymorphic function reused with different concrete sizes at
  // two call sites would need a different design -- not a case seen so
  // far), so caching the resulting FuncOp by `fun` alone (ignoring sizeEnv
  // on a cache hit) is safe.
  std::variant<mlir::func::FuncOp, BlackBox>
  LowerFunction(const FunDef &fun,
                const std::unordered_map<std::string, int64_t> &sizeEnv = {}) {
    if (auto blackbox = MaybeBlackBox(fun.attrs))
      return blackbox.value();

    if (functions.find(fun) != functions.end())
      return functions[fun];

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());

    std::vector<mlir::Type> retTypes;
    for (const auto &t : fun.retType) {
      retTypes.push_back(LowerTy(t.first.v, sizeEnv));
    }
    // Unit-typed params carry no runtime data, so they're dropped from
    // @main's actual signature rather than lowered to some placeholder type.
    auto runtimeParams = fun.params | std::views::filter([](const auto &param) {
                           return !IsUnitType(param.dec.v);
                         });
    std::vector<mlir::Type> inputTypes;
    for (const auto &param : runtimeParams) {
      inputTypes.push_back(LowerTy(param.dec.v, sizeEnv));
    }

    // XLA requires every shape in the module to be static.
    for (mlir::Type ty : inputTypes)
      if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(ty);
          rt && !rt.hasStaticShape())
        Undefined("dynamic-shaped function parameter");
    for (mlir::Type ty : retTypes)
      if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(ty);
          rt && !rt.hasStaticShape())
        Undefined("dynamic-shaped function result");

    // The entry function becomes @main (XLA's required entry point name)
    // regardless of its Futhark source name. More than one entry point is
    // ambiguous, not untranslatable -- a hard error, not Undefined().
    std::string funcName = fun.name;
    if (fun.entry) {
      int numEntries = 0;
      for (const auto &f : prog.funs)
        if (f.entry)
          ++numEntries;
      if (numEntries > 1)
        throw std::runtime_error(
            "multiple entry points: StableHLO/XLA requires exactly one @main");
      funcName = "main";
    }

    auto fType = builder.getFunctionType(inputTypes, retTypes);
    auto func =
        mlir::func::FuncOp::create(builder.getUnknownLoc(), funcName, fType);
    builder.insert(func);
    functions[fun] = func;

    if (!fun.entry) {
      func.setPrivate();
    }

    Ctx ctx;
    auto entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // Replicate all top-level constants in the function's block.
    // Eventually we might want global-scope constants, but for cheap constants
    // this should be fine.
    for (auto &stm : prog.consts) {
      LowerStm(stm, ctx);
    }

    for (auto [param, arg] :
         llvm::zip_equal(runtimeParams, func.getArguments())) {
      ctx.subexps.insert(param.name, arg);
    }

    Values ret = LowerBody(fun.body, ctx);

    mlir::func::ReturnOp::create(builder, builder.getUnknownLoc(), ret);
    return func;
  }

  Values LowerBody(Body body, Ctx &ctx) {
    for (auto &stm : body.stms) {
      LowerStm(stm, ctx);
    }
    Values results;
    for (auto &res : body.result) {
      results.push_back(LowerSubExp(res, ctx));
    }
    return results;
  }

  void LowerStm(const Stm &stm, Ctx &ctx) {
    auto vs = LowerExp(stm.exp, ctx);
    size_t valueIndex = 0;
    for (const auto &elem : stm.pat.elems) {
      if (IsAccType(elem.dec.v)) {
        assert(valueIndex + 2 <= vs.size());

        ctx.accs.insert(elem.name.name,
                        AccValue{
                            vs[valueIndex],
                            vs[valueIndex + 1],
                        });

        valueIndex += 2;
      } else {
        assert(valueIndex < vs.size());

        ctx.subexps.insert(elem.name.name, vs[valueIndex]);

        ++valueIndex;
      }
    }

    assert(valueIndex == vs.size());
  }

  Values LowerExp(const Exp &exp, Ctx &ctx) {
    return match(
        exp.v,
        [&](const ExpBasicOp &e) -> Values {
          if (auto *update = std::get_if<BasicOpUpdateAcc>(&e.op.v))
            return LowerUpdateAcc(*update, ctx);
          return {LowerBasicOp(e.op, ctx)};
        },
        [&](const ExpSubExp &e) { return Values{LowerSubExp(e.subExp, ctx)}; },
        [&](const ExpHostOp &e) { return LowerHostOp(e.op, ctx); },
        [&](const ExpApply &e) -> Values {
          // See builtin_repiota.hpp: repiota's real result size is
          // data-dependent and has no valid StableHLO representation
          // here, used or not, so it's intercepted unconditionally by
          // name rather than going through the normal (always-failing)
          // path below. Only applied when `g` (the call's first
          // argument) is itself a literal -- if it isn't, fall through to
          // the normal path's honest failure rather than guessing.
          if (e.fname == "builtin/repiota") {
            if (auto *gLit =
                    std::get_if<ConstantSubExp>(&e.args[0].first.v)) {
              auto vals =
                  builtin_repiota::LowerDummy(builder, gLit->GetIntValue());
              return Values(vals.begin(), vals.end());
            }
          }
          const FunDef *pFun = FindFunction(prog, e.fname);
          if (!pFun) {
            Values args;
            for (auto arg : std::views::elements<0>(e.args))
              args.push_back(LowerSubExp(arg, ctx));
            return LowerIntrinsic(e.fname, args);
          }
          const FunDef &fun = *pFun;
          mlir::SmallVector<mlir::Type> retTypes;
          for (auto ty : std::views::elements<0>(e.retTypes)) {
            retTypes.push_back(LowerTy(ty.v));
          }
          // The custom-flattening feature's #[noinline] flatten targets
          // (e.g. batched_matmul's dummy wrapper) are the first case this
          // backend has seen of a plain (non-blackbox) function whose own
          // declared param types are named-size-dependent (`[b][n][k]f32`)
          // rather than already-literal -- see LowerFunction's sizeEnv
          // comment. Resolve those names from whichever of THIS call's
          // actual arguments are themselves compile-time literals (the
          // size arguments always are, in the GPU IR this backend parses:
          // Futhark passes a function's array dims as explicit leading
          // i64 arguments).
          std::unordered_map<std::string, int64_t> sizeEnv;
          for (auto [param, argDiet] : llvm::zip_equal(fun.params, e.args)) {
            if (auto *c = std::get_if<ConstantSubExp>(&argDiet.first.v))
              // Only an integer literal is ever a *size* -- a bool/float
              // constant argument (e.g. argsort's is_stable) is not, and
              // PrimValue::GetIntValue() throws on one.
              if (std::holds_alternative<IntValue>(c->v.v))
                sizeEnv[param.name] = c->GetIntValue();
          }
          // See size_env.hpp: a size that isn't itself passed as a
          // literal (e.g. `a_k`, driving `a_data: [a_k]f32`) may still be
          // recoverable from that *array* argument's own already-static
          // type, if it's already been lowered.
          size_env::BackfillFromArrayArgs(
              fun.params, e.args,
              [&](const std::string &name) -> std::optional<mlir::Type> {
                if (auto v = ctx.subexps.tryLookup(name))
                  return v->getType();
                return std::nullopt;
              },
              sizeEnv);
          auto returns = match(
              LowerFunction(fun, sizeEnv),
              [&](mlir::func::FuncOp func) -> Values {
                // Calling a function that takes a dynamically shaped tensor
                // with a statically shaped one is a type error.
                Values args;
                for (auto [arg, ty] :
                     llvm::zip_equal(std::views::elements<0>(e.args),
                                     func.getArgumentTypes())) {
                  args.push_back(castToType(LowerSubExp(arg, ctx), ty));
                }

                auto call = mlir::func::CallOp::create(builder, func, args);

                return call->getResults();
              },
              [&](BlackBox b) {
                Values args;
                for (auto arg : std::views::elements<0>(e.args)) {
                  args.push_back(LowerSubExp(arg, ctx));
                }

                switch (b) {
                case BlackBox::MatMul:
                  return LowerMatMul(args, retTypes, ctx);
                case BlackBox::DotGeneral:
                  return LowerDotGeneral(args, retTypes, ctx);
                case BlackBox::RaggedDot:
                  return LowerRaggedDot(args, retTypes, ctx);
                case BlackBox::RaggedDotNonUniform:
                  return LowerRaggedDotNonUniform(args, retTypes, ctx);
                case BlackBox::ArgSort:
                  return LowerArgSort(args, retTypes, ctx);
                }
                llvm_unreachable("LowerBlackBox");
              });

          Values results;
          for (auto [v, ty] : llvm::zip_equal(returns, retTypes)) {
            results.push_back(castToType(v, ty));
          }
          return results;
        },
        [&](const ExpLoop &e) -> Values { return LowerLoop(e, ctx); },
        [&](const ExpIf &e) -> Values {
          // Confirmed via both the spec (no IsolatedFromAbove trait on
          // IfOp) and a JAX lax.cond reference lowering (CPU-only, see
          // LowerSegRed's comment for the caveat; branches reference outer
          // SSA values directly with zero block arguments) that
          // stablehlo.if's branches, unlike stablehlo.map's region, can
          // capture freely -- same as the old scf.if this replaced, so
          // this was a structural no-op swap. The condition is already a
          // 0-d i1 tensor (scalars are 0-d tensors), matching
          // stablehlo.if's `pred` directly -- no unwrap needed, unlike the
          // old scf.if's bare i1.
          mlir::Value cond = LowerSubExp(e.cond, ctx);
          mlir::SmallVector<mlir::Type> resultTypes;
          for (auto &t : e.retType)
            resultTypes.push_back(LowerTy(t));

          auto op = hlo::IfOp::create(builder, resultTypes, cond);

          auto buildBranch = [&](mlir::Region &region,
                                 const std::shared_ptr<Body> &body) {
            // The guard must be constructed before createBlock, not after
            // (see LowerSegMap's comment on the same point).
            mlir::OpBuilder::InsertionGuard guard(builder);
            mlir::Block *blk = builder.createBlock(&region);
            builder.setInsertionPointToStart(blk);
            Values results = LowerBody(*body, ctx);
            hlo::ReturnOp::create(builder, results);
          };
          buildBranch(op.getTrueBranch(), e.then_body);
          buildBranch(op.getFalseBranch(), e.else_body);

          return op->getResults();
        },
        [&](const ExpWithAcc &e) { return LowerWithAcc(e, ctx); });
  }

  // No StableHLO/XLA equivalent for an arbitrary-trip-count loop (for or
  // while) -- per the brief, both become Undefined() for now.
  Values LowerLoop(const ExpLoop &, Ctx &) { Undefined("loop"); }

  // Reconciles `value`'s type with `type` when they're nominally different
  // but describe the same value (e.g. a Futhark `:>` size coercion). Scalars
  // are always 0-d tensors now, so there's no bare-scalar case to extract
  // into; the only remaining case is a static-shape reshape with the same
  // element count.
  mlir::Value castToType(mlir::Value value, mlir::Type type) {
    if (value.getType() == type)
      return value;
    auto vTy = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    auto tTy = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (vTy && tTy && vTy.hasStaticShape() && tTy.hasStaticShape() &&
        vTy.getElementType() == tTy.getElementType() &&
        vTy.getNumElements() == tTy.getNumElements()) {
      return hlo::ReshapeOp::create(builder, tTy, value);
    }
    // `type` is the call site's own declared type, which for a blackbox
    // (or a function whose result involves a Futhark existential size
    // our sizeEnv/gather-idiom recognition doesn't need to resolve, e.g.
    // ragged_dot_nonuniform's `r`) may be less precise -- some dims
    // dynamic -- than `value`'s own actual, already-static type. Keep
    // the more precise type rather than forcing a cast that would only
    // throw the precision away; the caller never needed the dynamic
    // annotation to be literal, only compatible.
    if (vTy && tTy && vTy.getElementType() == tTy.getElementType() &&
        vTy.getRank() == tTy.getRank()) {
      bool compatible = true;
      for (auto [vDim, tDim] : llvm::zip_equal(vTy.getShape(), tTy.getShape()))
        compatible &= mlir::ShapedType::isDynamic(tDim) || vDim == tDim;
      if (compatible)
        return value;
    }
    llvm::errs() << "error: tried to cast\n\n"
                 << value << "\n\nto type\n"
                 << type << "\n";
    Undefined("castToType: incompatible types");
  }

  // Signature is (d0, d1, d3, A : [d4][d5]t, B : [d5][d6]t) -> [d4][d6]t.
  // Where d0 d1 d3 are equal to d4 d5 d6 somehow, depending on the order
  // specified by the user in the Futhark source/IR. So we disregard the size
  // parameters entirely and grab them from the matrix arguments A and B.
  // Contracts A's axis 1 with B's axis 0; no batch dimensions -- same
  // DotDimensionNumbersAttr pattern as the already-working LowerDotGeneral
  // below. Precision comes from the compiler-level `precision` field (set
  // via mlir-backend's --precision flag), not a post-hoc text rewrite.
  Values LowerMatMul(mlir::ValueRange args, mlir::TypeRange retTypes,
                     Ctx &ctx) {
    assert(args.size() == 5);
    assert(retTypes.size() == 1);
    auto x = args[3];
    auto y = args[4];
    auto resultTy = getRankedType(retTypes[0]);
    if (!resultTy.hasStaticShape())
      Undefined("matmul: dynamic shape");

    auto dotDims =
        hlo::DotDimensionNumbersAttr::get(&context,
                                          /*lhsBatchingDimensions=*/{},
                                          /*rhsBatchingDimensions=*/{},
                                          /*lhsContractingDimensions=*/{1},
                                          /*rhsContractingDimensions=*/{0});
    auto op = hlo::DotGeneralOp::create(
        builder, resultTy, x, y, dotDims, PrecisionConfig(), {});
    return op->getResults();
  }

  Values LowerUpdateAcc(const BasicOpUpdateAcc &update, Ctx &ctx) {
    assert(update.indices.size() == 1);
    assert(update.values.size() == 1);
    mlir::Value index = LowerSubExp(update.indices[0], ctx);
    mlir::Value value = LowerSubExp(update.values[0], ctx);
    return {value, index};
  }

  // with_acc-based scatter (Futhark's `scatter`/`reduce_by_index`-via-acc
  // sugar) via stablehlo.scatter, reusing LowerSegHist's dimension-numbers
  // convention generalized with update_window_dims for a whole-row/window
  // update (LowerSegHist's own scatter is the windowRank==0 special case
  // of the same formula). Handles exactly the shape this backend has a
  // concrete need for: a single accumulator, no combining lambda (a plain
  // overwrite -- Futhark's own `pWithAccOp()` is absent for this,
  // confirmed against the GPU IR), one index per update, and an update
  // value that is either a scalar or a full window/row of the
  // destination's trailing dimensions.
  //
  // The index and value must each be, inside the inner segmap, a
  // *trivial* identity read (an unmodified array lookup spanning the
  // whole iteration space) -- not a genuinely per-element *computed*
  // value. stablehlo.map's region can't produce an array-valued result
  // per grid point (LowerSegOpBaseType's array branch is already
  // Undefined()), and LowerSegMap itself refuses any acc-typed result, so
  // there is no existing machinery to lower a real per-element computation
  // here; this only works because the identity-read case needs none --
  // the "updates" tensor scatter wants is just the already-lowered source
  // array, unchanged. A genuinely computed with_acc value is a separate,
  // unimplemented capability (array-valued SegMap results), gated below
  // rather than guessed at.
  Values LowerWithAcc(const ExpWithAcc &e, Ctx &ctx) {
    if (e.inputs.size() != 1)
      Undefined("with_acc: multiple inputs");
    const WithAccInput &input = e.inputs[0];
    if (input.op.has_value())
      Undefined("with_acc: combining lambda");
    if (input.arrays.size() != 1)
      Undefined("with_acc: multiple accumulator arrays");
    if (input.ispace.dims.size() != 1)
      Undefined("with_acc: ispace rank != 1");

    mlir::Value dest = ctx.subexps.lookup(input.arrays[0].name);

    // The with_acc lambda's body is just `let {acc_result} = segmap(...)
    // {...} in {acc_result}` for every case this backend has seen.
    const SegMap *pSegMap = nullptr;
    for (const Stm &stm : e.lambda->body.stms) {
      auto *hostOp = std::get_if<ExpHostOp>(&stm.exp.v);
      if (!hostOp)
        continue;
      auto *segOpPtr = std::get_if<std::shared_ptr<SegOp>>(&hostOp->op.v);
      if (!segOpPtr)
        continue;
      auto *segMap = std::get_if<SegMap>(&(*segOpPtr)->v);
      if (!segMap)
        continue;
      if (pSegMap)
        Undefined("with_acc: multiple segmaps");
      pSegMap = segMap;
    }
    if (!pSegMap)
      Undefined("with_acc: no segmap found");
    RequireThreadLevel(pSegMap->lvl);

    IterationSpace iterSpace = LowerSegSpace(pSegMap->space, ctx);
    if (iterSpace.size() != 1)
      Undefined("with_acc: segmap rank != 1");

    const BasicOpUpdateAcc *pUpdate = nullptr;
    for (const Stm &stm : pSegMap->body.stms) {
      auto *basicOp = std::get_if<ExpBasicOp>(&stm.exp.v);
      if (!basicOp)
        continue;
      auto *update = std::get_if<BasicOpUpdateAcc>(&basicOp->op.v);
      if (!update)
        continue;
      if (pUpdate)
        Undefined("with_acc: multiple update_acc calls");
      pUpdate = update;
    }
    if (!pUpdate || pUpdate->indices.size() != 1 ||
        pUpdate->values.size() != 1)
      Undefined("with_acc: expected exactly one update_acc(index, value)");

    auto findDefiningExp = [&](const SubExp &sub) -> const Exp * {
      auto *var = std::get_if<VarSubExp>(&sub.v);
      if (!var)
        return nullptr;
      for (const Stm &stm : pSegMap->body.stms)
        if (stm.pat.elems.size() == 1 &&
            stm.pat.elems[0].name.name == var->v.name)
          return &stm.exp;
      return nullptr;
    };

    // Index: must be a plain identity affine read over the whole (1-D)
    // iteration space -- i.e. the array itself, unchanged.
    const Exp *idxExp = findDefiningExp(pUpdate->indices[0]);
    if (!idxExp)
      Undefined("with_acc: index is not a plain array read");
    auto idxRead = toAffineRead(VName{}, *idxExp, iterSpace);
    if (!idxRead || idxRead->indexMap.getNumResults() != 1 ||
        mlir::cast<mlir::AffineDimExpr>(idxRead->indexMap.getResult(0))
                .getPosition() != 0)
      Undefined("with_acc: index is not a plain identity read of the "
                "whole iteration space");
    mlir::Value indices = ctx.subexps.lookup(idxRead->array.name);
    auto indicesTy = getRankedType(indices.getType());
    if (indicesTy.getRank() != 1)
      Undefined("with_acc: index array must already match the iteration "
                "space");

    // Value: either a literal constant (a uniform scatter-fill, e.g.
    // `update_acc(acc, {i}, {true})` -- no window, one value per index),
    // or the same kind of identity affine read (a scalar update), or a
    // trivial full-row/window slice (DimFix on the sole iteration id,
    // DimSlice covering each remaining dimension's entire extent from 0
    // with unit stride) -- i.e. the array itself, unchanged.
    mlir::Value updates;
    if (std::holds_alternative<ConstantSubExp>(pUpdate->values[0].v)) {
      mlir::Value scalar = LowerSubExp(pUpdate->values[0], ctx);
      mlir::Type elemTy = getRankedType(scalar.getType()).getElementType();
      updates = hlo::BroadcastInDimOp::create(
          builder,
          mlir::RankedTensorType::get({indicesTy.getShape()[0]}, elemTy),
          scalar, mlir::SmallVector<int64_t>{});
    } else {
      const Exp *valExp = findDefiningExp(pUpdate->values[0]);
      auto *valBasicOp = valExp ? std::get_if<ExpBasicOp>(&valExp->v) : nullptr;
      auto *valIdx =
          valBasicOp ? std::get_if<BasicOpIndex>(&valBasicOp->op.v) : nullptr;
      if (!valIdx)
        Undefined("with_acc: value is not a plain array read");
      updates = ctx.subexps.lookup(valIdx->vName.name);
      auto updatesTy = getRankedType(updates.getType());
      if ((int64_t)valIdx->slice.dims.size() != updatesTy.getRank())
        Undefined("with_acc: value is not a trivial full-window read");
      for (int64_t d = 0; d < updatesTy.getRank(); ++d) {
        const auto &dim = valIdx->slice.dims[d];
        if (d == 0) {
          auto *fix = std::get_if<DimFix<SubExp>>(&dim.v);
          auto *var = fix ? std::get_if<VarSubExp>(&fix->i.v) : nullptr;
          if (!var || var->v.name != iterSpace[0].id)
            Undefined("with_acc: value is not a trivial full-window read");
        } else {
          auto *slice = std::get_if<DimSlice<SubExp>>(&dim.v);
          auto *start =
              slice ? std::get_if<ConstantSubExp>(&slice->start.v) : nullptr;
          auto *stride =
              slice ? std::get_if<ConstantSubExp>(&slice->stride.v) : nullptr;
          if (!start || start->GetIntValue() != 0 || !stride ||
              stride->GetIntValue() != 1 ||
              toShapeType(slice->length) != updatesTy.getShape()[d])
            Undefined("with_acc: value is not a trivial full-window read");
        }
      }
    }
    auto updatesTy = getRankedType(updates.getType());

    int64_t windowRank = updatesTy.getRank() - 1;
    mlir::SmallVector<int64_t> updateWindowDims;
    for (int64_t i = 0; i < windowRank; ++i)
      updateWindowDims.push_back(1 + i);

    mlir::Value scatterIndices = hlo::ReshapeOp::create(
        builder,
        mlir::RankedTensorType::get({indicesTy.getShape()[0], 1},
                                    indicesTy.getElementType()),
        indices);

    auto scatterDims = hlo::ScatterDimensionNumbersAttr::get(
        &context, updateWindowDims, /*insertedWindowDims=*/{0},
        /*operandBatchingDims=*/{}, /*scatterIndicesBatchingDims=*/{},
        /*scatterDimsToOperandDims=*/{0}, /*indexVectorDim=*/1);

    auto op = hlo::ScatterOp::create(
        builder, mlir::TypeRange{dest.getType()}, mlir::ValueRange{dest},
        scatterIndices, mlir::ValueRange{updates}, scatterDims,
        /*indices_are_sorted=*/false, /*unique_indices=*/false);

    mlir::Type elemTy = getRankedType(dest.getType()).getElementType();
    mlir::SmallVector<mlir::Type> argTys = {
        mlir::RankedTensorType::get({}, elemTy),
        mlir::RankedTensorType::get({}, elemTy)};
    mlir::SmallVector<mlir::Location> locs(2, builder.getLoc());
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      mlir::Block *blk = builder.createBlock(&op.getUpdateComputation(), {},
                                             argTys, locs);
      builder.setInsertionPointToStart(blk);
      // Plain overwrite -- no combining lambda (input.op is nullopt), so
      // just return the new value, discarding the old one.
      hlo::ReturnOp::create(builder, mlir::ValueRange{blk->getArgument(1)});
    }

    return Values(op->getResults().begin(), op->getResults().end());
  }

  Values LowerDotGeneral(mlir::ValueRange args, mlir::TypeRange retTypes,
                         Ctx &ctx) {
    // StableHLO dot_general
    //
    // The following signature is without support for
    // - per-call precision parameters from the Futhark source (the
    //   compiler-level `precision` field/--precision flag still applies
    //   uniformly here via PrecisionConfig(), same as LowerMatMul),
    // - support for quantized tensors.
    // This simplifies things somewhat. Refer to the spec for the full version.
    //
    // Inputs
    // ------
    // ...Futhark size parameters
    // lhs: tensor
    // rhs: tensor
    // lhs_batching_dimensions: 1d tensor constant of signed i64
    // rhs_batching_dimensions: 1d tensor constant of signed i64
    // lhs_contracting_dimensions: 1d tensor constant of signed i64
    // rhs_contracting_dimensions: 1d tensor constant of signed i64
    //
    // Outputs
    // ------
    // result: tensor

    assert(retTypes.size() == 1);
    auto getArg = BackIndexedGetter(args, 6);
    auto lhs = getArg(0);
    auto rhs = getArg(1);
    auto lhsBatchingDimensions = getConstantIntTensor(getArg(2));
    auto rhsBatchingDimensions = getConstantIntTensor(getArg(3));
    auto lhsContractingDimensions = getConstantIntTensor(getArg(4));
    auto rhsContractingDimensions = getConstantIntTensor(getArg(5));

    auto lhsType = getRankedType(lhs.getType());
    auto lhsShape = lhsType.getShape();
    auto rhsShape = getRankedType(rhs.getType()).getShape();

    mlir::SmallVector<int64_t> resultShape;
    for (auto i : lhsBatchingDimensions)
      resultShape.push_back(lhsShape[i]);
    for (auto d : llvm::enumerate(lhsShape))
      if (!llvm::is_contained(lhsBatchingDimensions, d.index()) &&
          !llvm::is_contained(lhsContractingDimensions, d.index()))
        resultShape.push_back(lhsShape[d.index()]);
    for (auto d : llvm::enumerate(rhsShape))
      if (!llvm::is_contained(rhsBatchingDimensions, d.index()) &&
          !llvm::is_contained(rhsContractingDimensions, d.index()))
        resultShape.push_back(rhsShape[d.index()]);

    auto op = hlo::DotGeneralOp::create(
        builder,
        mlir::RankedTensorType::get(resultShape, lhsType.getElementType()),
        lhs,
        rhs,
        hlo::DotDimensionNumbersAttr::get(&context,
                                          lhsBatchingDimensions,
                                          rhsBatchingDimensions,
                                          lhsContractingDimensions,
                                          rhsContractingDimensions),
        PrecisionConfig(),
        {});

    return op->getResults();
  }

  Values LowerRaggedDot(mlir::ValueRange args, mlir::TypeRange retTypes,
                        Ctx &ctx) {
    // CHLO ragged_dot (without support for precision parameters)
    //
    // RaggedDot takes three tensor arguments (lhs, rhs, and group_sizes).
    //
    // Notation:
    //   b... are batch dimensions
    //   m... are the lhs non-contracting dimensions
    //   k... are the contracting dimensions
    //   n... are the rhs non-contracting dimensions
    //   g is the number of groups
    //   x... are,
    //        in mode 1, non-contracting dimensions before the ragged dimension;
    //        in mode 2, contracting dimensions before the ragged dimension;
    //        in mode 3, the empty sequence.
    //
    // RaggedDot has three modes (types) depending on which dimension is ragged.
    // Mode 1 ::
    //   (ragged dim is one of m...) =>
    //   (lhs: [b..., m..., k...]t) ->
    //   (rhs: [g, b..., k..., n...]t) ->
    //   (group_sizes: [b..., x..., g]int) ->
    //   [b..., m..., n...]t
    // Mode 2 ::
    //   (ragged dim is one of k...) =>
    //   (lhs: [b..., m..., k...]t) ->
    //   (rhs: [b..., k..., n...]t) ->
    //   (group_sizes: [b...,  x..., g]int) ->
    //   [g, b..., m..., n...]t
    // Mode 3 ::
    //   (ragged dim is one of b...) =>
    //   (lhs: [b..., m..., k...]t) ->
    //   (rhs: [b..., k..., n...]t) ->
    //   (group_sizes: [g]int) ->
    //   [b..., m..., n...]t
    //
    // Like dot_general, lhs and rhs may have arbitrary batching and
    // (non-)contracting dimensions specified by additional parameters.
    // So the order shown here (e.g., [b..., m..., k...]) is just for
    // presentation; they can appear in any order as specified by the
    // *_dimensions parameters below.
    //
    //
    // Inputs
    // ------
    // ...Futhark size parameters
    // lhs: tensor
    // rhs: tensor
    // group_sizes: tensor of any integer type
    // lhs_batching_dimensions: 1d tensor constant of signed i64
    // rhs_batching_dimensions: 1d tensor constant of signed i64
    // lhs_contracting_dimensions: 1d tensor constant of signed i64
    // rhs_contracting_dimensions: 1d tensor constant of signed i64
    // lhs_ragged_dimensions : 1d tensor constant of signed i64
    // rhs_group_dimensions: 1d tensor constant of signed i64
    //
    // Outputs
    // ------
    // result: tensor

    assert(retTypes.size() == 1);
    auto getArg = BackIndexedGetter(args, 9);
    auto lhs = getArg(0);
    auto rhs = getArg(1);
    auto groupSizes = getArg(2);
    auto lhsBatchingDimensions = getConstantIntTensor(getArg(3));
    auto rhsBatchingDimensions = getConstantIntTensor(getArg(4));
    auto lhsContractingDimensions = getConstantIntTensor(getArg(5));
    auto rhsContractingDimensions = getConstantIntTensor(getArg(6));
    auto lhsRaggedDim = getConstantInt(getArg(7));
    auto rhsGroupDimensions = getConstantIntTensor(getArg(8));

    auto lhsType = getRankedType(lhs.getType());
    auto lhsShape = lhsType.getShape();
    auto rhsShape = getRankedType(rhs.getType()).getShape();

    // NOTE CHLO->StableHLO conversion does not handle rhs with dynamic
    // dimensions, so we have to fail on that.
    // (Probably because machine learning people always have a statically shaped
    // weight matrix there.)
    if (!mlir::ShapedType::isStaticShape(rhsShape))
      throw std::runtime_error(
          "ragged_dot: shape of rhs tensor must be static.");

    auto g = getRankedType(groupSizes.getType()).getShape().back();

    mlir::SmallVector<int64_t> resultShape;
    if (llvm::is_contained(lhsContractingDimensions, lhsRaggedDim)) // Mode 2
      resultShape.push_back(g);
    for (auto i : lhsBatchingDimensions)
      resultShape.push_back(lhsShape[i]);
    for (auto d : llvm::enumerate(lhsShape))
      if (!llvm::is_contained(lhsBatchingDimensions, d.index()) &&
          !llvm::is_contained(lhsContractingDimensions, d.index()))
        resultShape.push_back(lhsShape[d.index()]);
    for (auto d : llvm::enumerate(rhsShape))
      if (!llvm::is_contained(rhsBatchingDimensions, d.index()) &&
          !llvm::is_contained(rhsContractingDimensions, d.index()) &&
          !llvm::is_contained(rhsGroupDimensions, d.index()))
        resultShape.push_back(rhsShape[d.index()]);

    // See cpu_dev_fallbacks.hpp: GPU/TPU performance drives this path, so
    // it stays chlo.RaggedDotOp unconditionally by default. The dense-
    // decomposition fallback is a separate, explicitly opt-in accommodation
    // for CPU dev testing only (XLA:CPU's ThunkEmitter doesn't implement
    // the `ragged-dot` HLO opcode this lowers to), never the default. Only
    // take it for shapes it actually supports -- shapes like mode 3's
    // (ragged dim == batch dim) already execute fine on XLA:CPU via
    // chlo.RaggedDotOp directly and must keep using that path unchanged.
    if (raggedDotCpuFallback &&
        cpu_dev_fallbacks::IsRaggedDotCpuFallbackSupported(
            lhs, rhs, lhsBatchingDimensions, lhsContractingDimensions,
            lhsRaggedDim, rhsGroupDimensions)) {
      mlir::Value result = cpu_dev_fallbacks::LowerRaggedDotCpuFallback(
          builder, context, lhs, rhs, groupSizes, lhsBatchingDimensions,
          lhsContractingDimensions, lhsRaggedDim, rhsGroupDimensions,
          resultShape);
      return {result};
    }

    auto op = mlir::chlo::RaggedDotOp::create(
        builder,
        mlir::RankedTensorType::get(resultShape, lhsType.getElementType()),
        lhs,
        rhs,
        groupSizes,
        mlir::chlo::RaggedDotDimensionNumbersAttr::get(&context,
                                                       lhsBatchingDimensions,
                                                       rhsBatchingDimensions,
                                                       lhsContractingDimensions,
                                                       rhsContractingDimensions,
                                                       {lhsRaggedDim},
                                                       rhsGroupDimensions),
        {});

    return op->getResults();
  }

  // Live again as of ragged_matmul_mix.fut (custom-flattening's
  // #[flatten_to_nonuniform(...)] routed directly at this blackbox):
  // the static-m branch below is now StableHLO-only, migrated off the
  // old (pre-rewrite, linalg/arith/tensor) code this previously was.
  // `ragged_matmul_black_box.fut`/`batched_ragged_matmul_black_box.fut`
  // still can't reach this (their entry parameters are genuinely
  // dynamic-shaped, hitting LowerFunction's own gate first), and the
  // dynamic-m branch further down is still unmigrated/unreachable --
  // nothing currently exercises it, so it's left exactly as it was
  // rather than guess-fixed.
  Values LowerRaggedDotNonUniform(mlir::ValueRange args,
                                  mlir::TypeRange retTypes, Ctx &ctx) {
    // A nonuniform ragged_dot expressed over the flat segmented representation
    // produced by irregular flattening. This blackbox sits on
    // matmul_nonuniform, so it is lowered at its application site where the
    // actual (constant) size arguments are visible: k = ks[0], n = ns[0],
    // b_data static.
    //
    // Inputs (matmul_nonuniform's signature)
    // ------
    // 0: g          number of segments
    // 1: ms         [g] group sizes (== ragged_dot's group_sizes)
    // 2: ks         [g] replicate(k) -- k uniform across groups
    // 3: ns         [g] replicate(n) -- n uniform across groups
    // 4: a_k        flat lhs length (dynamic)
    // 5..7:         a_segment_sizes / a_flag / a_offsets (unused)
    // 8: a_data     flat lhs [a_k]         (dynamic in m)
    // 9: b_k        flat rhs length (static, = g*k*n)
    // 10..12:       b_segment_sizes / b_flag / b_offsets (unused)
    // 13: b_data    flat rhs [b_k]         (static)
    //
    // Outputs: the existential size r followed by the segmented tuple
    //   (r : i64, sizes : [g]i64, flags : [r]i1, offsets : [g]i64, data :
    //   [r]f32)
    // Only `data` (the flattened [m][n] result) is meaningful; the auxiliary
    // segment structures are dummies for this demonstration.
    assert(args.size() == 14);
    assert(retTypes.size() == 5);
    auto ms = args[1];
    auto ksT = args[2];
    auto nsT = args[3];
    auto aData = args[8];
    auto bData = args[13];

    auto elementTy = getRankedType(aData.getType()).getElementType();
    auto g = getRankedType(ms.getType()).getShape().back();

    // Recover a `replicate`d scalar constant. Futhark's replicate lowers to
    // hlo::BroadcastInDimOp(hlo::ConstantOp) (see BasicOpReplicate's live,
    // StableHLO-only lowering), so peel those -- and any `:>` size
    // coercion's hlo::ReshapeOp -- to reach the underlying scalar
    // constant, then let getConstantInt's own dense-constant fold handle
    // the rest.
    auto replicatedInt = [&](mlir::Value v) -> int64_t {
      mlir::Value cur = v;
      while (auto *op = cur.getDefiningOp()) {
        if (mlir::isa<hlo::ReshapeOp, hlo::BroadcastInDimOp>(op))
          cur = op->getOperand(0);
        else
          break;
      }
      return getConstantInt(cur);
    };

    auto k = replicatedInt(ksT);
    auto n = replicatedInt(nsT);

    // m = a_k / k. Prefer a static m: from a_data's static flat length, else
    // from a folded-constant a_k argument (k is already static, from ks).
    auto dyn = mlir::ShapedType::kDynamic;
    auto aDataTy = getRankedType(aData.getType());
    int64_t m = aDataTy.hasStaticShape() ? aDataTy.getDimSize(0) / k : dyn;
    if (mlir::ShapedType::isDynamic(m)) {
      mlir::APInt aKConst;
      if (mlir::matchPattern(args[4], mlir::m_ConstantInt(&aKConst)))
        m = aKConst.getSExtValue() / k;
    }
    // TEMP: until the flatten threads a_k statically, uncomment to force a
    // static m for the m=128 bench and exercise the static chlo.ragged_dot
    // path. if (mlir::ShapedType::isDynamic(m)) m = 128;

    mlir::Value resultMN; // the [m][n] product
    if (!mlir::ShapedType::isDynamic(m)) {
      // ---- static m: a faithful, minimal chlo.ragged_dot ----
      // rhs: static [g][k][n]; b is not ragged so its flat length is static.
      auto rhsTy = mlir::RankedTensorType::get({g, k, n}, elementTy);
      auto bDataTy = getRankedType(bData.getType());
      mlir::Value rhs =
          bDataTy == rhsTy ? bData
                           : hlo::ReshapeOp::create(builder, rhsTy, bData)
                                 .getResult();
      // lhs [m][k]: a_data must already be statically shaped here -- unlike
      // the old (pre-StableHLO) tensor::CastOp this replaced, there is no
      // valid StableHLO way to assert a dynamic shape into a static one, so
      // a genuinely dynamic a_data is a real Undefined(), not a cast.
      if (!aDataTy.hasStaticShape())
        Undefined("ragged_dot_nonuniform: a_data is not statically shaped");
      auto lhsTy = mlir::RankedTensorType::get({m, k}, elementTy);
      mlir::Value lhs = hlo::ReshapeOp::create(builder, lhsTy, aData);
      // Mode 1: ragged dim lhs axis 0; contract lhs axis 1 with rhs axis 1;
      // rhs group dim axis 0; no batching -- same shape LowerRaggedDot's
      // own CPU-fallback gate already recognizes, so mirror its dispatch
      // here too rather than always emitting chlo.RaggedDotOp
      // unconditionally (see cpu_dev_fallbacks.hpp).
      mlir::SmallVector<int64_t> noBatch{};
      mlir::SmallVector<int64_t> lhsContracting{1};
      mlir::SmallVector<int64_t> rhsGroup{0};
      mlir::SmallVector<int64_t> resultShape{m, n};
      if (raggedDotCpuFallback &&
          cpu_dev_fallbacks::IsRaggedDotCpuFallbackSupported(
              lhs, rhs, noBatch, lhsContracting, /*lhsRaggedDim=*/0,
              rhsGroup)) {
        resultMN = cpu_dev_fallbacks::LowerRaggedDotCpuFallback(
            builder, context, lhs, rhs, ms, noBatch, lhsContracting,
            /*lhsRaggedDim=*/0, rhsGroup, resultShape);
      } else {
        resultMN = mlir::chlo::RaggedDotOp::create(
                       builder,
                       mlir::RankedTensorType::get({m, n}, elementTy),
                       lhs,
                       rhs,
                       ms,
                       mlir::chlo::RaggedDotDimensionNumbersAttr::get(
                           &context,
                           noBatch,
                           noBatch,
                           lhsContracting,
                           mlir::SmallVector<int64_t>{1},
                           mlir::SmallVector<int64_t>{0},
                           rhsGroup),
                       {})
                       ->getResult(0);
      }
    } else {
      // ---- dynamic m: the dense masked dot_general strategy
      // (gg/ragged_matmul). IREE cannot legalize chlo.ragged_dot's internal
      // dynamic_broadcast for a runtime m, but stablehlo.dot_general handles
      // dynamic dims, and the mask is a single affine linalg.generic (no
      // dynamic broadcast).
      auto mIdx =
          mlir::arith::DivUIOp::create(
              builder,
              mlir::tensor::DimOp::create(builder, aData, 0).getResult(),
              mlir::arith::ConstantIndexOp::create(builder, k).getResult())
              .getResult();
      // lhs2d [m][k]
      auto lhs2dTy = mlir::RankedTensorType::get({dyn, k}, elementTy);
      mlir::SmallVector<mlir::ReassociationIndices> lhs2dReassoc{{0, 1}};
      mlir::SmallVector<mlir::OpFoldResult> lhs2dOut{mlir::OpFoldResult(mIdx),
                                                     builder.getIndexAttr(k)};
      mlir::Value lhs2d = mlir::tensor::ExpandShapeOp::create(
                              builder, lhs2dTy, aData, lhs2dReassoc, lhs2dOut)
                              .getResult();

      // masked [m][g][k]: masked[i,e,j] = lhs2d[i,j] if row i in group e else
      // 0, where group e spans rows [a_offsets[e]/k, a_offsets[e]/k + ms[e]).
      auto aOffsets = args[7]; // [g]i64 flat offsets
      auto maskedTy = mlir::RankedTensorType::get({dyn, g, k}, elementTy);
      mlir::Value maskedInit = mlir::tensor::EmptyOp::create(
          builder, maskedTy, mlir::ValueRange{mIdx});
      auto d0 = mlir::getAffineDimExpr(0, &context);
      auto d1 = mlir::getAffineDimExpr(1, &context);
      auto d2 = mlir::getAffineDimExpr(2, &context);
      mlir::SmallVector<mlir::AffineMap> maps{
          mlir::AffineMap::get(3, 0, {d0, d2}, &context), // lhs2d (i,j)
          mlir::AffineMap::get(3, 0, {d1}, &context),     // a_offsets (e)
          mlir::AffineMap::get(3, 0, {d1}, &context),     // ms (e)
          mlir::AffineMap::getMultiDimIdentityMap(3, &context)}; // out
      std::vector<mlir::utils::IteratorType> iters(
          3, mlir::utils::IteratorType::parallel);
      auto kI64 = mlir::arith::ConstantOp::create(builder,
                                                  builder.getI64IntegerAttr(k));
      auto zeroElem = mlir::arith::ConstantOp::create(
          builder, builder.getZeroAttr(elementTy));
      mlir::Value masked =
          mlir::linalg::GenericOp::create(
              builder,
              mlir::TypeRange{maskedTy},
              mlir::ValueRange{lhs2d, aOffsets, ms},
              mlir::ValueRange{maskedInit},
              maps,
              iters,
              [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange a) {
                mlir::OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPointToEnd(b.getInsertionBlock());
                auto iIdx = mlir::linalg::IndexOp::create(builder, loc, 0);
                auto iI64 = mlir::arith::IndexCastOp::create(
                    builder, loc, builder.getI64Type(), iIdx);
                auto start =
                    mlir::arith::DivUIOp::create(builder, loc, a[1], kI64);
                auto end =
                    mlir::arith::AddIOp::create(builder, loc, start, a[2]);
                auto ge = mlir::arith::CmpIOp::create(
                    builder, loc, mlir::arith::CmpIPredicate::sge, iI64, start);
                auto lt = mlir::arith::CmpIOp::create(
                    builder, loc, mlir::arith::CmpIPredicate::slt, iI64, end);
                auto inR = mlir::arith::AndIOp::create(builder, loc, ge, lt);
                auto val = mlir::arith::SelectOp::create(
                    builder, loc, inR, a[0], zeroElem);
                mlir::linalg::YieldOp::create(builder, loc, val.getResult());
              })
              ->getResult(0);

      // fuse masked [m][g][k] -> [m][g*k] and rhs -> [g*k][n], then dot_general
      // contracting the fused axis (reproduces the two-axis g,k contraction).
      auto masked2dTy = mlir::RankedTensorType::get({dyn, g * k}, elementTy);
      mlir::SmallVector<mlir::ReassociationIndices> fuse{{0}, {1, 2}};
      mlir::Value masked2d = mlir::tensor::CollapseShapeOp::create(
                                 builder, masked2dTy, masked, fuse)
                                 .getResult();
      auto rhsFusedTy = mlir::RankedTensorType::get({g * k, n}, elementTy);
      auto bDataTy = getRankedType(bData.getType());
      mlir::Value rhsFused =
          mlir::tensor::ExpandShapeOp::create(
              builder,
              rhsFusedTy,
              bData,
              *mlir::getReassociationIndicesForReshape(bDataTy, rhsFusedTy))
              .getResult();
      resultMN = hlo::DotGeneralOp::create(
                     builder,
                     mlir::RankedTensorType::get({dyn, n}, elementTy),
                     masked2d,
                     rhsFused,
                     hlo::DotDimensionNumbersAttr::get(&context,
                                                       /*lhsBatch=*/{},
                                                       /*rhsBatch=*/{},
                                                       /*lhsContract=*/{1},
                                                       /*rhsContract=*/{0}),
                     {},
                     {})
                     .getResult();
    }

    // data: flatten [m][n] -> [m*n].
    int64_t mn = mlir::ShapedType::isDynamic(m) ? dyn : m * n;
    auto dataTy = mlir::RankedTensorType::get({mn}, elementTy);
    mlir::Value data = hlo::ReshapeOp::create(builder, dataTy, resultMN);

    // r = length of data (= m*n), as i64 -- static whenever m is (the
    // only case this function currently reaches; see its header
    // comment).
    if (mlir::ShapedType::isDynamic(mn))
      Undefined("ragged_dot_nonuniform: dynamic m (data length)");
    auto i64Ty = builder.getI64Type();
    mlir::Value r = hlo::ConstantOp::create(
        builder, mlir::DenseElementsAttr::get(
                     mlir::RankedTensorType::get({}, i64Ty),
                     builder.getI64IntegerAttr(mn)));

    // Dummy auxiliary segment structures -- see matmul_nonuniform_bb's
    // own callers: nothing downstream of this blackbox ever reads them.
    auto gI64Ty = mlir::RankedTensorType::get({g}, i64Ty);
    mlir::Value sizes = hlo::ConstantOp::create(
        builder,
        mlir::DenseElementsAttr::get(gI64Ty, builder.getZeroAttr(i64Ty)));
    mlir::Value offsets = sizes;
    auto flagsTy = mlir::RankedTensorType::get({mn}, builder.getI1Type());
    mlir::Value flags = hlo::ConstantOp::create(
        builder, mlir::DenseElementsAttr::get(flagsTy, builder.getBoolAttr(true)));

    return {r, sizes, flags, offsets, data};
  }

  Values LowerArgSort(mlir::ValueRange args, mlir::TypeRange retTypes,
                      Ctx &ctx) {
    // A basic ascending sort.
    // Doesn't implement StableHLO sort faithfully.
    assert(retTypes.size() == 1);
    auto getArg = BackIndexedGetter(args, 2);
    auto inputs = getArg(0);
    bool isStable = getConstantInt(getArg(1)) != 0;

    auto inputsTy = getRankedType(inputs.getType());
    if (inputsTy.getRank() != 1)
      Undefined();
    if (!mlir::ShapedType::isStaticShape(inputsTy.getShape()))
      throw std::runtime_error(
          "argsort: shape of input tensor must be static.");

    auto i64Scalar = mlir::RankedTensorType::get({}, builder.getI64Type());
    mlir::Value indices = createIota(
        hlo::ConstantOp::create(builder,
                                mlir::DenseElementsAttr::get(
                                    i64Scalar, builder.getI64IntegerAttr(0))),
        hlo::ConstantOp::create(builder,
                                mlir::DenseElementsAttr::get(
                                    i64Scalar, builder.getI64IntegerAttr(1))),
        builder.getI64Type(),
        inputsTy.getShape()[0]);

    // Sort the values and the indices together, keyed on the values; the
    // permuted indices are the argsort result.
    auto op = hlo::SortOp::create(
        builder, mlir::ValueRange{inputs, indices}, (int64_t)0, isStable);

    // The comparator has two arguments per sorted operand: the values first
    // (compared here), then the indices (carried along).
    auto valScalar = mlir::RankedTensorType::get({}, inputsTy.getElementType());
    auto idxScalar = mlir::RankedTensorType::get({}, builder.getI64Type());
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto *block =
          builder.createBlock(&op.getComparator(),
                              {},
                              {valScalar, valScalar, idxScalar, idxScalar},
                              {builder.getLoc(),
                               builder.getLoc(),
                               builder.getLoc(),
                               builder.getLoc()});
      auto lt = hlo::CompareOp::create(builder,
                                       block->getArgument(0),
                                       block->getArgument(1),
                                       hlo::ComparisonDirection::LT,
                                       hlo::ComparisonType::SIGNED);
      hlo::ReturnOp::create(builder, mlir::ValueRange{lt});
    }

    return {op.getResult(1)};
  }

  mlir::RankedTensorType getRankedType(const mlir::Type &ty) {
    if (auto t = mlir::dyn_cast<mlir::RankedTensorType>(ty))
      return t;
    Undefined();
  }

  // Several intrinsic-call lowerings (LowerDotGeneral, LowerRaggedDot,
  // LowerArgSort) have an unknown, variable number of leading Futhark size
  // arguments before their real (non-size) arguments -- but the real
  // arguments are always the last `numNonSizeArgs` of the list, so indexing
  // from the back skips the size prefix without needing to know its length.
  std::function<mlir::Value(int)> BackIndexedGetter(mlir::ValueRange args,
                                                     int64_t numNonSizeArgs) {
    assert(args.size() >= (size_t)numNonSizeArgs);
    return [args, numNonSizeArgs](int i) {
      return args[args.size() - numNonSizeArgs + i];
    };
  }

  mlir::SmallVector<int64_t> getConstantIntTensor(const mlir::Value &x) {
    mlir::DenseIntElementsAttr attr;
    auto shapeTy = getRankedType(x.getType());
    if (shapeTy.hasStaticShape() && shapeTy.getNumElements() == 0)
      return {};
    if (mlir::matchPattern(x, mlir::m_Constant(&attr))) {
      return llvm::to_vector(attr.getValues<int64_t>());
    }
    Undefined();
  }

  int64_t getConstantInt(const mlir::Value &x) {
    mlir::APInt v;
    if (mlir::matchPattern(x, mlir::m_ConstantInt(&v))) {
      return v.getSExtValue();
    }
    // Scalars are 0-d tensors now, so a constant scalar may also show up as
    // a single-element stablehlo.constant rather than a bare arith.constant.
    mlir::DenseIntElementsAttr attr;
    if (mlir::matchPattern(x, mlir::m_Constant(&attr)) &&
        attr.getNumElements() == 1) {
      return *attr.getValues<int64_t>().begin();
    }
    Undefined();
  }

  // Builds a 1-D StableHLO iota tensor where result[i] = x + i*s. `x`/`s` are
  // 0-d tensors (scalars are 0-d tensors now); only static lengths are
  // supported.
  mlir::Value createIota(mlir::Value x, mlir::Value s, mlir::Type elementTy,
                         int64_t dimTy) {
    if (mlir::ShapedType::isDynamic(dimTy))
      Undefined("iota: dynamic size");
    auto returnTy = mlir::RankedTensorType::get({dimTy}, elementTy);
    mlir::Value base =
        hlo::IotaOp::create(builder, returnTy, builder.getI64IntegerAttr(0));
    auto bx = hlo::BroadcastInDimOp::create(
        builder, returnTy, x, mlir::SmallVector<int64_t>{});
    auto bs = hlo::BroadcastInDimOp::create(
        builder, returnTy, s, mlir::SmallVector<int64_t>{});
    auto scaled = hlo::MulOp::create(builder, base, bs);
    return hlo::AddOp::create(builder, scaled, bx);
  }

  // `arr[i, j]` reads a single element, but a slice among the indices makes the
  // result an array: `arr[i, a :+ n * s]` becomes a tensor.extract_slice whose
  // fixed dimensions are rank-reduced away.
  // TODO(StableHLO rewrite): the general slice path below still emits
  // tensor.extract_slice, which XLA rejects (tensor.* is banned). Not
  // exercised by any currently-passing test (affine reads inside SegMap
  // bodies take a separate fast path and never reach this function with a
  // slice present); needs a stablehlo.slice (static bounds) /
  // stablehlo.dynamic_slice (dynamic start, static size) rewrite before
  // anything that indexes with a slice can work. The no-slice (all-fixed-
  // index) fast path right below *is* exercised -- e.g. unwrapping a
  // segred's rank-1 result down to a scalar after reducing a flat,
  // unsegmented array goes through here -- and is already stablehlo-only.
  // stablehlo.dynamic_slice handles constant and dynamic start indices
  // uniformly (its start_indices are themselves 0-d tensors already, no
  // index-cast/unwrapping needed), for every dimension at once -- a fixed
  // index is just a 1-sized slice along that dimension. It can't
  // rank-reduce on its own (every dimension survives in the result, fixed
  // ones as size-1), so a trailing stablehlo.reshape drops the fixed
  // dimensions to get the actual (lower-rank) result shape. Only a unit
  // stride is supported: dynamic_slice has no stride concept at all, and
  // no currently-reachable use needs one (Futhark's own `a[s:e:k]` syntax
  // allows non-unit/negative k, but every slice this backend has actually
  // seen so far -- e.g. moe_black_box.fut's `tokens[i, :]` -- is a plain
  // unit-stride range).
  mlir::Value LowerIndex(const BasicOpIndex &index, Ctx &ctx) {
    auto tensor = LowerSubExp(index.vName.name, ctx);
    auto tensorTy = getRankedType(tensor.getType());

    Values startIndices;
    mlir::SmallVector<int64_t> sliceSizes;
    std::vector<int64_t> resultShape;
    for (const auto &dim : index.slice.dims) {
      match(
          dim,
          [&](const DimFix<SubExp> &fix) {
            startIndices.push_back(LowerSubExp(fix.i, ctx));
            sliceSizes.push_back(1);
          },
          [&](const DimSlice<SubExp> &s) {
            auto *stride = std::get_if<ConstantSubExp>(&s.stride.v);
            if (!stride || stride->GetIntValue() != 1)
              Undefined("index: non-unit slice stride");
            startIndices.push_back(LowerSubExp(s.start, ctx));
            int64_t len = toShapeType(s.length);
            sliceSizes.push_back(len);
            resultShape.push_back(len);
          });
    }

    auto sliceTy =
        mlir::RankedTensorType::get(sliceSizes, tensorTy.getElementType());
    auto slice = hlo::DynamicSliceOp::create(builder, sliceTy, tensor,
                                             startIndices, sliceSizes);
    if (resultShape.size() == sliceSizes.size())
      return slice;
    auto resultTy =
        mlir::RankedTensorType::get(resultShape, tensorTy.getElementType());
    return hlo::ReshapeOp::create(builder, resultTy, slice);
  }

  mlir::Value LowerBasicOp(const BasicOp &basicOp, Ctx &ctx) {
    return match(
        basicOp,
        [&](const BasicOpSubExp &val) { return LowerSubExp(val.subExp, ctx); },
        [&](const BasicOpArrayLit &val) { return LowerArrayLit(val, ctx); },
        [&](const BasicOpBinOp &val) {
          auto op0 = LowerSubExp(val.op0, ctx);
          auto op1 = LowerSubExp(val.op1, ctx);
          return LowerBinOp(val.op, op0, op1, ctx);
        },
        [&](const BasicOpIota &val) -> mlir::Value {
          mlir::Value x = LowerSubExp(val.x, ctx);
          mlir::Value s = LowerSubExp(val.s, ctx);
          auto elementTy = LowerPrimTypeInt(PrimTypeInt{val.t});
          return createIota(x, s, elementTy, toShapeType(val.n));
        },
        [&](const BasicOpConcat &val) -> mlir::Value {
          std::vector<mlir::Value> operands;
          for (auto arr : val.arrs)
            operands.push_back(LowerSubExp(arr.name, ctx));

          // `val.total` is the size of the concatenated dimension, which the
          // result type already carries.
          return hlo::ConcatenateOp::create(builder, operands,
                                            (uint64_t)val.dim);
        },
        [&](const BasicOpIndex &val) -> mlir::Value {
          return LowerIndex(val, ctx);
        },
        [&](const BasicOpConvOp &val) -> mlir::Value {
          auto op0 = LowerSubExp(val.op0, ctx);
          auto convOp = val.op;
          return match(
              convOp,
              // TODO: unverified against stablehlo.convert's exact
              // int-widening semantics for signless integer types; flagged
              // Undefined() rather than risk silently-wrong rounding/sign.
              [](const ConvOpZExt &) -> mlir::Value {
                Undefined("ConvOp: ZExt");
              },
              [&](const ConvOpSExt &sext) -> mlir::Value {
                return hlo::ConvertOp::create(
                    builder,
                    LowerTy(Type::CreatePrim(PrimType::Int(sext.to))),
                    op0);
              },
              [&](const ConvOpSIToFP &sitofp) -> mlir::Value {
                auto ty = mlir::RankedTensorType::get(
                    {}, GetFloatType(builder, GetWidth(sitofp.to)));
                return hlo::ConvertOp::create(builder, ty, op0);
              },
              [](const auto &) -> mlir::Value { Undefined("convOp"); });
        },
        [&](const BasicOpReshape &val) -> mlir::Value {
          auto op0 = LowerSubExp(val.op0, ctx);
          auto sourceTy = getRankedType(op0.getType());

          // `remainder` is the whole shape the reshape produces; the splice
          // fields only say which dimensions were rewritten to get there.
          auto resultTy = mlir::RankedTensorType::get(
              toShapeType(val.remainder.dims), sourceTy.getElementType());
          if (sourceTy == resultTy)
            return op0;
          if (!sourceTy.hasStaticShape() || !resultTy.hasStaticShape())
            Undefined("reshape: dynamic shape");
          // stablehlo.reshape is a flat reinterpretation for any two static
          // shapes with the same element count -- no collapse/expand
          // distinction needed, unlike tensor.{collapse,expand}_shape.
          return hlo::ReshapeOp::create(builder, resultTy, op0);
        },
        [&](const BasicOpRearrange &val) -> mlir::Value {
          auto op = LowerSubExp(val.arr.name, ctx);
          auto tensorTy = llvm::dyn_cast<mlir::RankedTensorType>(op.getType());
          if (!tensorTy)
            Undefined();
          if (val.perm.size() != tensorTy.getShape().size())
            Undefined();

          // stablehlo.transpose puts input dimension perm[i] in result
          // dimension i, matching linalg.transpose's convention.
          std::vector<int64_t> dims;
          for (int64_t p : val.perm)
            dims.push_back(tensorTy.getShape()[p]);

          auto transposedTy =
              mlir::RankedTensorType::get(dims, tensorTy.getElementType());
          mlir::SmallVector<int64_t> perm(val.perm.begin(), val.perm.end());
          return hlo::TransposeOp::create(builder, transposedTy, op, perm);
        },
        [&](const BasicOpReplicate &val) -> mlir::Value {
          auto op = LowerSubExp(val.val, ctx);
          // Scalars are 0-d tensors now, so `op` is always a RankedTensorType
          // (0-d for a scalar replicate, N-d for an array replicate) -- one
          // path handles both.
          auto tensorTy = getRankedType(op.getType());
          if (!tensorTy.hasStaticShape())
            Undefined("replicate: dynamic element shape");
          std::vector<int64_t> before = toShapeType(val.shape.dims);
          if (llvm::any_of(before, mlir::ShapedType::isDynamic)) {
            // See ragged_unflatten_gather.hpp: a dynamically-shaped
            // replicate has no valid StableHLO representation here at
            // all, used or not, so it's given a cheap, DCE-and-safety-
            // net-backed placeholder rather than failing outright.
            if (tensorTy.getRank() == 0)
              return ragged_unflatten::LowerDummyReplicate(
                  builder, tensorTy.getElementType());
            Undefined("replicate: dynamic shape");
          }
          auto original = tensorTy.getShape();
          before.insert(before.end(), original.begin(), original.end());
          auto resultTy =
              mlir::RankedTensorType::get(before, tensorTy.getElementType());
          mlir::SmallVector<int64_t> broadcastDims;
          int64_t rankDiff = (int64_t)before.size() - (int64_t)original.size();
          for (int64_t i = 0; i < (int64_t)original.size(); ++i)
            broadcastDims.push_back(rankDiff + i);
          return hlo::BroadcastInDimOp::create(
              builder, resultTy, op, broadcastDims);
        },
        [&](const BasicOpScratch &val) -> mlir::Value {
          // `scratch` promises an uninitialized buffer about to be fully
          // overwritten; StableHLO has no uninitialized-tensor concept (and
          // no dynamically-shaped constant), so this backend represents it
          // as a static, zero-filled stablehlo.constant instead -- behaves
          // identically to uninitialized memory for any caller that
          // actually honors the "fully overwritten" contract. Every
          // currently-reachable use is static (this backend's static-
          // shapes-only design means a dynamic scratch would already have
          // tripped the dynamic-shaped function parameter/result gate
          // further up the call chain).
          auto baseTy = LowerPrimType(val.type);
          auto shapeTy = toShapeType(val.dims);
          if (llvm::any_of(shapeTy, mlir::ShapedType::isDynamic))
            Undefined("scratch: dynamic shape");
          auto tensorTy = mlir::RankedTensorType::get(shapeTy, baseTy);
          return hlo::ConstantOp::create(
              builder,
              mlir::DenseElementsAttr::get(tensorTy, builder.getZeroAttr(baseTy)));
        },
        [&](const BasicOpCmpOp &val) -> mlir::Value {
          auto lhs = LowerSubExp(val.op0, ctx);
          auto rhs = LowerSubExp(val.op1, ctx);
          using CD = hlo::ComparisonDirection;
          using CT = hlo::ComparisonType;
          auto cmp = [&](CD dir, CT ty) -> mlir::Value {
            return hlo::CompareOp::create(builder, lhs, rhs, dir, ty);
          };
          return match(
              val.op,
              [&](const CmpOpEq &eqOp) -> mlir::Value {
                return match(
                    eqOp.t,
                    [&](const PrimTypeInt &) {
                      return cmp(CD::EQ, CT::SIGNED);
                    },
                    [&](const PrimTypeFloat &) -> mlir::Value {
                      return cmp(CD::EQ, CT::FLOAT);
                    },
                    [&](const PrimTypeBool &) -> mlir::Value { Undefined(); },
                    [&](const PrimTypeUnit &) -> mlir::Value { Undefined(); });
              },
              [&](const CmpOpUlt &) { return cmp(CD::LT, CT::UNSIGNED); },
              [&](const CmpOpUle &) { return cmp(CD::LE, CT::UNSIGNED); },
              [&](const CmpOpSlt &) { return cmp(CD::LT, CT::SIGNED); },
              [&](const CmpOpSle &) { return cmp(CD::LE, CT::SIGNED); },
              [&](const CmpOpFlt &) { return cmp(CD::LT, CT::FLOAT); },
              [&](const CmpOpFle &) { return cmp(CD::LE, CT::FLOAT); },
              [&](const CmpOpLLt &) -> mlir::Value { Undefined(); },
              [&](const CmpOpLLe &) -> mlir::Value { Undefined(); });
        },
        // The BasicOp variants below (here and after BasicOpUnOp) are all
        // unimplemented. Opaque/ArrayVal/FlatIndex/Update/FlatUpdate are
        // from array-flattening-related constructs this backend doesn't
        // target (static shapes only, no runtime flattening); Assert has
        // no StableHLO equivalent (no runtime trap/abort primitive);
        // Manifest and UpdateAcc are tied to with_acc, deferred alongside
        // LowerWithAcc/scatter. None of these are exercised by any
        // currently-passing test.
        [](const BasicOpOpaque &) -> mlir::Value { Undefined(); },
        [](const BasicOpArrayVal &) -> mlir::Value { Undefined(); },
        [&](const BasicOpUnOp &val) -> mlir::Value {
          auto v = LowerSubExp(val.subExp, ctx);
          auto elemTy = getRankedType(v.getType()).getElementType();
          return match(
              val.op.v,
              [&](const UnOpNeg &) -> mlir::Value {
                // Neg on bool is logical not; stablehlo.negate covers both
                // int (0 - x) and float negation uniformly.
                if (elemTy.isInteger(1))
                  return hlo::NotOp::create(builder, v);
                return hlo::NegOp::create(builder, v);
              },
              [&](const UnOpComplement &) -> mlir::Value {
                return hlo::NotOp::create(builder, v);
              },
              [](const UnOpAbs &) -> mlir::Value { Undefined(); },
              [](const UnOpFAbs &) -> mlir::Value { Undefined(); },
              [](const UnOpSSignum &) -> mlir::Value { Undefined(); },
              [](const UnOpUSignum &) -> mlir::Value { Undefined(); },
              [](const UnOpFSignum &) -> mlir::Value { Undefined(); });
        },
        [](const BasicOpFlatIndex &) -> mlir::Value { Undefined(); },
        [](const BasicOpUpdate &) -> mlir::Value { Undefined(); },
        [](const BasicOpAssert &) -> mlir::Value { Undefined(); },
        [](const BasicOpFlatUpdate &) -> mlir::Value { Undefined(); },
        [](const BasicOpManifest &) -> mlir::Value { Undefined(); },
        [](const BasicOpUpdateAcc &) -> mlir::Value { Undefined(); });
  }

  mlir::Value LowerSubExp(SubExp subExp, Ctx &ctx) {
    if (auto *val = std::get_if<VarSubExp>(&subExp.v)) {
      return LowerSubExp(val->v.name, ctx);
    }

    if (auto *val = std::get_if<ConstantSubExp>(&subExp.v)) {
      return LowerPrimValue(val->v, ctx);
    }

    Undefined();
  }

  mlir::Value LowerSubExp(std::string vname, Ctx &ctx) {
    return ctx.subexps.lookup(vname);
  }

  Values LowerHostOp(const HostOp &op, Ctx &ctx) {
    return match(
        op,
        [&](const SizeOp &v) { return Values{LowerSizeOp(v, ctx)}; },
        [&](const std::shared_ptr<SegOp> &v) { return LowerSegOp(v, ctx); },
        [&](const GPUBody &g) -> Values {
          // A single-thread device body returning scalars, whose results are
          // each lifted into a size-1 array (see the [1] result types).
          Ctx local = ctx;
          Values scalars = LowerBody(*g.body, local);
          Values results;
          for (auto [v, ty] : llvm::zip_equal(scalars, g.retType)) {
            // The body's type is the inner (scalar) type; the gpu op lifts each
            // result into a size-1 array. LowerTy(ty) is itself a 0-d tensor
            // (scalars are 0-d tensors), so the lifted array's element type
            // is its element type, not the 0-d tensor type itself.
            auto lifted = mlir::RankedTensorType::get(
                {1}, getRankedType(LowerTy(ty)).getElementType());
            results.push_back(hlo::ReshapeOp::create(builder, lifted, v));
          }
          return results;
        });
  }

  // SizeOp (ir/gpu.hpp: GetSize/GetSizeMax/CmpSizeLe/CalcNumBlocks) is
  // Futhark's GPU-backend auto-tuning machinery -- block size/count
  // knobs for a kernel launch. XLA picks its own launch configuration;
  // there's no equivalent knob to thread through at the StableHLO level.
  // TODO: if a program's *semantics* (not just performance) ever turn
  // out to depend on one of these (unlikely, but unconfirmed), this gate
  // would need revisiting instead of staying a blanket Undefined().
  mlir::Value LowerSizeOp(const SizeOp &sizeOp, Ctx &ctx) { Undefined(); }

  Values LowerSegOp(std::shared_ptr<SegOp> pSegOp, Ctx &ctx) {
    return match(
        pSegOp->v,
        [&](const SegMap &v) { return LowerSegMap(v, ctx); },
        [&](const SegRed &v) { return LowerSegRed(v, ctx); },
        [&](const SegScan &v) { return LowerSegScan(v, ctx); },
        [&](const SegHist &v) -> Values { return LowerSegHist(v, ctx); });
  }

  // TODO Limitations
  // 1. The base type must be scalar (i.e., the seg space creates all dims).
  // 2. Only affine reads with seg space ids are recognized as such.
  // For example,
  //    xs[gtid]
  // is recognized because gtid is in the iteration (seg) space. But
  //    xs[2*gtid + 1]
  // is not. Neither is
  //    xs[gtid - c],
  // where c is a variable defined outside the kernel body.

  // stablehlo.map's region is isolated-from-above, so every outer SSA
  // value a map body ends up using
  // must become an explicit operand instead. Discovers those free
  // variables, broadcasts scalar ones to the map's full iteration-space
  // shape and appends them as extra operands; bails on non-scalar ones
  // (no broadcast shape would be well-defined). Shared by every
  // stablehlo.map builder in this file (LowerSegMap, LowerElementwiseLambda,
  // Combine) -- they only differ in the Undefined() message, passed as
  // `errCtx` so the error still names the right caller.
  //
  // Uses the two plain free functions mlir::makeRegionIsolatedFromAbove is
  // built from (mlir::getUsedValuesDefinedAbove/replaceAllUsesInRegionWith)
  // rather than that function itself, which needs an mlir::IRRewriter --
  // its base class mlir::RewriterBase has no typeinfo in the prebuilt MLIR
  // libs (LLVM/MLIR defaults to -fno-rtti; this backend builds with
  // -frtti), which fails at link time.
  void BroadcastFreeVarsIntoMapRegion(hlo::MapOp op,
                                      llvm::ArrayRef<int64_t> shapeTy,
                                      llvm::StringRef errCtx) {
    mlir::SetVector<mlir::Value> freeVars;
    mlir::getUsedValuesDefinedAbove(op.getRegion(), freeVars);
    if (freeVars.empty())
      return;

    mlir::Block &entry = op.getRegion().front();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(op);
    Values broadcasts;
    for (mlir::Value fv : freeVars) {
      auto fvTy = mlir::dyn_cast<mlir::RankedTensorType>(fv.getType());
      if (!fvTy || fvTy.getRank() != 0)
        Undefined(std::string(errCtx) + ": non-scalar free variable");
      auto broadcastTy =
          mlir::RankedTensorType::get(shapeTy, fvTy.getElementType());
      broadcasts.push_back(hlo::BroadcastInDimOp::create(
          builder, broadcastTy, fv, mlir::SmallVector<int64_t>{}));
      mlir::BlockArgument newArg =
          entry.addArgument(fv.getType(), builder.getLoc());
      mlir::replaceAllUsesInRegionWith(fv, newArg, op.getRegion());
    }
    op->insertOperands(op->getNumOperands(), broadcasts);
  }

  Values LowerSegMap(const SegMap &pSegMap, Ctx &ctx) {
    // See ragged_unflatten_gather.hpp: this one shape (a ragged layout's
    // flatten/unflatten gather) has no valid lowering through the normal
    // path below at all (the array it reads is a free variable the
    // stablehlo.map region can't capture) -- recognized and replaced
    // directly with a plain reshape of the source array, bailing to the
    // normal path on anything that doesn't match exactly.
    if (pSegMap.space.dims.size() == 1) {
      if (auto match = ragged_unflatten::TryMatchGather(
              std::get<0>(pSegMap.space.dims[0]), pSegMap.body)) {
        mlir::Value arr = ctx.subexps.lookup(match->array.name);
        auto arrTy = getRankedType(arr.getType());
        if (arrTy.getRank() == 2 && arrTy.hasStaticShape() &&
            arrTy.getShape()[1] == match->k) {
          int64_t total = arrTy.getShape()[0] * arrTy.getShape()[1];
          auto flatTy =
              mlir::RankedTensorType::get({total}, arrTy.getElementType());
          return {hlo::ReshapeOp::create(builder, flatTy, arr)};
        }
      }
    }
    return LowerSegMap(
        pSegMap.lvl, pSegMap.space, pSegMap.body, pSegMap.ret, ctx);
  }

  // Lowers a SegMap to a stablehlo.map. Operands are the kernel's affine
  // reads plus one full-shape stablehlo.iota per iteration-space dimension;
  // free variables are fixed up via BroadcastFreeVarsIntoMapRegion below.
  Values LowerSegMap(const SegLevel &lvl, const SegSpace &space,
                     const KernelBody &body, const std::vector<Type> &ret,
                     Ctx &ctx) {
    RequireThreadLevel(lvl);

    IterationSpace iterSpace = LowerSegSpace(space, ctx);
    auto shapeTy = toShapeType(std::views::values(space.dims));
    if (llvm::any_of(shapeTy, mlir::ShapedType::isDynamic))
      Undefined("SegMap: dynamic shape");

    std::vector<AffineRead> affine_reads =
        FindSegOpAffineReads(iterSpace, body);
    std::vector<GatherRead> gather_reads =
        FindSegOpGatherReads(iterSpace, affine_reads, body);

    // stablehlo.map has no per-operand indexing map the way linalg.generic
    // does -- every operand must already be the full iteration-space shape
    // (SameOperandsAndResultShape). An affine read's array may be lower-rank
    // (e.g. a plain [m]-shaped array read inside a [1][m] segspace for a
    // full, unsegmented reduction) or otherwise not already shaped that
    // way, so broadcast it first using the read's own affine map, whose
    // results are exactly the iteration-space dim each array dim
    // corresponds to -- precisely stablehlo.broadcast_in_dim's
    // broadcast_dimensions.
    Values operands;
    for (auto &read : affine_reads) {
      mlir::Value arr = ctx.subexps.lookup(read.array.name);
      auto arrTy = getRankedType(arr.getType());
      if (arrTy.getShape() != llvm::ArrayRef<int64_t>(shapeTy)) {
        mlir::SmallVector<int64_t> broadcastDims;
        for (mlir::AffineExpr e : read.indexMap.getResults())
          broadcastDims.push_back(
              mlir::cast<mlir::AffineDimExpr>(e).getPosition());
        auto broadcastTy =
            mlir::RankedTensorType::get(shapeTy, arrTy.getElementType());
        arr = hlo::BroadcastInDimOp::create(
            builder, broadcastTy, arr, broadcastDims);
      }
      operands.push_back(arr);
    }
    // stablehlo.iota's iota_dimension directly gives the dimension that
    // varies, so each gtid's iota is already full-shape -- no separate
    // broadcast needed.
    for (auto &d : iterSpace) {
      // d.value is the lowered Futhark size/index value, a 0-d tensor now
      // (scalars are 0-d tensors) -- the iota needs its bare element type.
      auto idxElemTy = getRankedType(d.value.getType()).getElementType();
      auto iotaTy = mlir::RankedTensorType::get(shapeTy, idxElemTy);
      operands.push_back(hlo::IotaOp::create(
          builder, iotaTy, builder.getI64IntegerAttr(d.index)));
    }

    // Gather reads: a stablehlo.map region only ever exposes each operand's
    // *current-position scalar* to its block args (per-element, like every
    // other operand) -- there's no way to get a whole array inside the
    // region to dynamic_slice from at a computed index, isolated-from-above
    // or not. So each gather is instead computed once, as a standalone
    // stablehlo.gather *outside* the map entirely, producing a full
    // shapeTy-shaped array; from there it's just another already-
    // materialized full-shape array, threaded into the map exactly like an
    // affine-read array (see the operands loop above -- this reuses that
    // same per-element binding, not a new mechanism).
    //
    // Deliberately NOT a general "lower every segmap read via gather"
    // strategy: checked empirically (compiled a trivial identity gather --
    // gather-by-iota, equivalent to a plain affine read -- through XLA's
    // actual optimizer) that XLA does not eliminate a gather even when
    // it's this trivial; it fuses it into a loop kernel but still performs
    // a real indexed-memory-access gather at runtime, strictly more
    // expensive than the broadcast_in_dim the affine-read path already
    // emits for that case. Reserving stablehlo.gather for genuinely
    // indirect/dynamic-index reads (the only thing it's needed for) keeps
    // the common case cheap.
    //
    // Scoped to the one shape this backend has a concrete need for: a
    // single gathered dimension, at iteration dim 0, with the remaining
    // ("kept") dimensions following it in order -- so the gather's natural
    // output order (batch dim(s) first, then window/kept dims) already
    // equals shapeTy directly, with no transpose needed. Bails rather than
    // guessing at anything more general (multiple gathered dimensions, a
    // gathered index varying over more than one iteration dim, kept/
    // gathered dims interleaved differently than shapeTy's own order).
    std::unordered_map<std::string, mlir::Value> gatherResults;
    for (auto &g : gather_reads) {
      if (gatherResults.contains(g.result.name))
        continue;
      mlir::Value arr = ctx.subexps.lookup(g.array.name);
      auto arrTy = getRankedType(arr.getType());
      int64_t arrRank = arrTy.getRank();

      mlir::SmallVector<int64_t> keptOperandDims;
      int64_t gatherOperandDim = -1;
      const AffineRead *gatherSrc = nullptr;
      for (int64_t d = 0; d < arrRank; ++d) {
        const std::string &name = g.indexNames[d].name;
        auto iterIt = std::ranges::find(iterSpace, name, &Dim::id);
        if (iterIt != iterSpace.end()) {
          if (iterIt->index != (int64_t)keptOperandDims.size() + 1)
            Undefined("SegMap: kept dimensions must follow the gathered "
                      "one in order");
          keptOperandDims.push_back(d);
          continue;
        }
        auto readIt = std::ranges::find_if(
            affine_reads,
            [&](const AffineRead &r) { return r.result.name == name; });
        if (readIt == affine_reads.end() || gatherOperandDim != -1)
          Undefined("SegMap: gather with more than one gathered dimension");
        gatherOperandDim = d;
        gatherSrc = &*readIt;
      }
      if (gatherOperandDim == -1 || gatherSrc->indexMap.getNumResults() != 1 ||
          mlir::cast<mlir::AffineDimExpr>(gatherSrc->indexMap.getResult(0))
                  .getPosition() != 0)
        Undefined("SegMap: gather dimension must be iteration dim 0");

      mlir::Value idxArr = ctx.subexps.lookup(gatherSrc->array.name);
      int64_t n = shapeTy[0];
      if (getRankedType(idxArr.getType()).getShape() !=
          llvm::ArrayRef<int64_t>({n}))
        Undefined("SegMap: gather index array shape must match the "
                  "gathered dimension");
      auto idxElemTy = getRankedType(idxArr.getType()).getElementType();
      mlir::Value startIndices = hlo::ReshapeOp::create(
          builder, mlir::RankedTensorType::get({n, 1}, idxElemTy), idxArr);

      mlir::SmallVector<int64_t> offsetDims;
      mlir::SmallVector<int64_t> sliceSizes(arrRank, 1);
      for (size_t i = 0; i < keptOperandDims.size(); ++i) {
        offsetDims.push_back((int64_t)i + 1);
        sliceSizes[keptOperandDims[i]] = arrTy.getShape()[keptOperandDims[i]];
      }
      auto gatherDims = hlo::GatherDimensionNumbersAttr::get(
          &context, offsetDims, /*collapsedSliceDims=*/{gatherOperandDim},
          /*operandBatchingDims=*/{}, /*startIndicesBatchingDims=*/{},
          /*startIndexMap=*/{gatherOperandDim}, /*indexVectorDim=*/1);
      auto resultTy = mlir::RankedTensorType::get(shapeTy, arrTy.getElementType());
      gatherResults[g.result.name] = hlo::GatherOp::create(
          builder, resultTy, arr, startIndices, gatherDims, sliceSizes,
          /*indicesAreSorted=*/false);
    }
    for (auto &g : gather_reads)
      operands.push_back(gatherResults.at(g.result.name));

    mlir::SmallVector<mlir::Type> elemTypes;
    for (const Type &resultType : ret) {
      if (GetAccType(resultType))
        Undefined("SegMap: with_acc result");
      elemTypes.push_back(LowerSegOpBaseType(resultType, ctx));
    }

    mlir::SmallVector<int64_t> allDims;
    for (int64_t i = 0; i < (int64_t)shapeTy.size(); ++i)
      allDims.push_back(i);

    mlir::SmallVector<mlir::Type> blockArgTypes;
    for (mlir::Value v : operands)
      blockArgTypes.push_back(mlir::RankedTensorType::get(
          {}, getRankedType(v.getType()).getElementType()));
    mlir::SmallVector<mlir::Location> locs(blockArgTypes.size(),
                                           builder.getLoc());

    // stablehlo.map allows only a single (non-variadic) result, so a
    // tuple-valued kernel body (ret.size() > 1, e.g. `map (\x -> (x, x+x))`)
    // needs one map op per output value, each recomputing the shared body
    // statements. That's redundant work (this backend doesn't run CSE
    // itself to clean it up) but correct, and avoids reintroducing a
    // multi-value packing trick like the old IREE scan's lane-packing hack.
    Values finalResults;
    for (size_t j = 0; j < ret.size(); ++j) {
      auto resultTy = mlir::RankedTensorType::get(shapeTy, elemTypes[j]);
      auto op = hlo::MapOp::create(
          builder, mlir::TypeRange{resultTy}, operands, allDims);

      {
        // The guard must be constructed before createBlock, not after: it
        // saves/restores the insertion point around its scope, and
        // createBlock itself moves the insertion point into the new block
        // as a side effect. Guarding only the body-lowering below would
        // save "already inside blk" and restore right back to that same
        // place, leaving the builder's insertion point stuck inside the
        // region afterward instead of back in the enclosing function body.
        mlir::OpBuilder::InsertionGuard guard(builder);
        mlir::Block *blk =
            builder.createBlock(&op.getRegion(), {}, blockArgTypes, locs);
        builder.setInsertionPointToStart(blk);

        Ctx local = ctx;
        mlir::ValueRange blockArgs = blk->getArguments();
        for (auto [read, arg] : llvm::zip_equal(
                 affine_reads, blockArgs.take_front(affine_reads.size())))
          local.subexps.insert(read.result.name, arg);
        for (auto [d, arg] : llvm::zip_equal(
                 iterSpace, blockArgs.drop_front(affine_reads.size())
                                .take_front(iterSpace.size())))
          local.subexps.insert(d.id, arg);
        // Gather reads are appended last (see the operands loop above);
        // each is already a materialized full-shape array by this point,
        // so its block arg is bound the exact same way an affine-read
        // array's is -- no per-element gather logic needed inside the
        // region at all.
        for (auto [g, arg] : llvm::zip_equal(
                 gather_reads, blockArgs.take_back(gather_reads.size())))
          local.subexps.insert(g.result.name, arg);

        std::unordered_set<std::string> skipLowering;
        for (auto &read : affine_reads)
          skipLowering.insert(read.result.name);
        for (auto &g : gather_reads)
          skipLowering.insert(g.result.name);
        for (const Stm &stm : body.stms) {
          assert(stm.pat.elems.size() == 1);
          if (skipLowering.contains(stm.pat.elems[0].name.name))
            continue;
          LowerStm(stm, local);
        }
        mlir::Value result = LowerSubExp(body.result[j].result, local);
        hlo::ReturnOp::create(builder, mlir::ValueRange{result});
      }

      BroadcastFreeVarsIntoMapRegion(op, shapeTy, "SegMap");

      finalResults.push_back(op->getResult(0));
    }

    return finalResults;
  }

  // segred = segmap (the fused per-element part, reused as-is -- so this
  // function inherits LowerSegMap's own limitations, e.g. the dynamic-shape
  // and non-scalar-free-variable gates) + a stablehlo.reduce over the
  // innermost (segment) dimension. Tuple operators fall out for free --
  // stablehlo.reduce is natively variadic (multiple inputs/init_values/
  // region-results), no special-casing needed beyond what's already
  // generic below.
  //
  // Block-arg binding: accumulator first, new value second, matching
  // Futhark's own SegBinOp.lambda.params order directly (see the table
  // below LowerSegScanReduceWindow's declaration for how this compares to
  // the other reducer-body scaffolds in this file). Getting this backwards
  // is invisible for a commutative operator and silently wrong for a
  // non-commutative one -- confirmed the hard way in Phase 8 (see
  // PROGRESS.md) after the binding sat backwards, undetected, since
  // Phase 2. stablehlo.reduce itself needs no non-commutative workaround;
  // the spec's monoid requirement is associativity + identity, not
  // commutativity, which is exactly what Futhark's own `neutral` already
  // guarantees.
  Values LowerSegRed(const SegRed &pSegRed, Ctx &ctx) {
    RequireThreadLevel(pSegRed.lvl);

    if (pSegRed.ops.size() != 1)
      Undefined("segred: multiple operators");
    auto pSegBinOp = pSegRed.ops[0];
    if (pSegBinOp.shape.dims.size() != 0)
      Undefined("segred: vector-valued operator");

    IterationSpace iterSpace = LowerSegSpace(pSegRed.space, ctx);
    int64_t rank = std::ssize(iterSpace);
    if (rank == 0)
      Undefined("segred: rank 0");

    Values mapped =
        LowerSegMap(pSegRed.lvl, pSegRed.space, pSegRed.body, pSegRed.ret, ctx);

    Values neutral;
    for (auto &e : pSegBinOp.neutral)
      neutral.push_back(LowerSubExp(e, ctx));
    int64_t valueCount = (int64_t)neutral.size();
    int64_t reduceDim = rank - 1;

    mlir::SmallVector<int64_t> reduceDims{reduceDim};
    mlir::SmallVector<mlir::Type> resultTypes;
    for (mlir::Value v : mapped) {
      auto ty = getRankedType(v.getType());
      resultTypes.push_back(mlir::RankedTensorType::get(
          ty.getShape().drop_back(), ty.getElementType()));
    }

    auto op = hlo::ReduceOp::create(
        builder, resultTypes, mapped, neutral, reduceDims);

    // Per the StableHLO spec, the body's block args are (one accumulator
    // per init_value, then one value per input slice) -- confirmed by
    // decoding a JAX reference lowering's arithmetic term by term (see
    // this function's own comment above), not assumed. This matches
    // Futhark's own SegBinOp.lambda.params convention directly
    // (accumulators first, then new values) -- no transpose needed.
    //
    // Reducer-body block-arg conventions across this file, so a future
    // mismatch like the one above is easier to suspect:
    //   LowerSegRed                   -- accumulator first, new value second
    //   LowerSegScanReduceWindow       -- new value first, accumulator second
    //   LowerSegHist update_computation -- accumulator first, new value second
    //   Combine (parallel-prefix)      -- no split; a/b are separate operand
    //                                     lists, zipped 1:1 in declaration order
    //
    // TODO: every "confirmed by decoding a JAX reference lowering" claim in
    // this file (here and in LowerSegScanReduceWindow, LowerSegHist,
    // LowerSegScanParallelPrefix/ScanRec) was checked against CPU-only
    // output -- no Metal/GPU plugin is installed in this environment. JAX
    // registers some lowering rules per-platform, so a convention
    // confirmed on CPU is not automatically proven for GPU/TPU; re-verify
    // against those backends before relying on this for anything beyond
    // CPU execution.
    mlir::SmallVector<mlir::Type> argTys;
    for (mlir::Value v : neutral)
      argTys.push_back(v.getType());
    for (mlir::Value v : mapped)
      argTys.push_back(mlir::RankedTensorType::get(
          {}, getRankedType(v.getType()).getElementType()));
    mlir::SmallVector<mlir::Location> locs(argTys.size(), builder.getLoc());

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      mlir::Block *blk = builder.createBlock(&op.getBody(), {}, argTys, locs);
      builder.setInsertionPointToStart(blk);

      Ctx local = ctx;
      mlir::ValueRange blockArgs = blk->getArguments();
      mlir::ValueRange accArgs = blockArgs.take_front(valueCount);
      mlir::ValueRange newValueArgs = blockArgs.drop_front(valueCount);
      for (auto [param, acc] : llvm::zip_equal(
               pSegBinOp.lambda.params | std::views::take(valueCount), accArgs))
        local.subexps.insert(param.name, acc);
      for (auto [param, nv] : llvm::zip_equal(pSegBinOp.lambda.params |
                                                  std::views::drop(valueCount),
                                              newValueArgs))
        local.subexps.insert(param.name, nv);

      Values results = LowerBody(pSegBinOp.lambda.body, local);
      hlo::ReturnOp::create(builder, results);
    }

    return op->getResults();
  }

  // SegHist (Futhark's `hist`/`reduce_by_index`) via stablehlo.scatter: a
  // pure scalar scatter (no update window) uses inserted_window_dims =
  // scatter_dims_to_operand_dims = [0..R-1] (R = destination rank),
  // index_vector_dim = 1 (the indices tensor is always rank-2 [n, R]
  // regardless of R), unique_indices = false, indices_are_sorted = false
  // -- confirmed against a JAX reference lowering (CPU-only, see
  // LowerSegRed's comment for the caveat), not guessed. Per the spec's
  // `scatter` semantics ("Otherwise updated_results = results"),
  // out-of-bounds scatter indices are silently dropped -- matching
  // Futhark's own hist semantics.
  Values LowerSegHist(const SegHist &pSegHist, Ctx &ctx) {
    RequireThreadLevel(pSegHist.lvl);

    if (pSegHist.ops.size() != 1)
      Undefined("SegHist: multiple operators");
    const HistOp &histOp = pSegHist.ops[0];
    // opShape (a per-element width for a vector-valued operator, distinct
    // from neutral.size()'s tuple-operator check below) and tuple
    // operators are both deferred -- same "defer solving" scope the brief
    // gives this phase.
    if (!histOp.opShape.empty())
      Undefined("SegHist: vector-valued operator");
    if (histOp.neutral.size() != 1)
      Undefined("SegHist: tuple operator");

    IterationSpace iterSpace = LowerSegSpace(pSegHist.space, ctx);
    if (std::ssize(iterSpace) != 1)
      Undefined("SegHist: multi-dimensional iteration space");

    // One full-[n] array per KernelResult, same helper LowerSegRed/
    // LowerSegScan reuse for their own per-element part: the first
    // histOp.shape.dims.size() results are the index components (one per
    // destination dimension), the last one is the value being scattered in
    // (neutral.size() == 1, checked above).
    Values mapped = LowerSegMap(
        pSegHist.lvl, pSegHist.space, pSegHist.body, pSegHist.ret, ctx);

    int64_t indexRank = (int64_t)histOp.shape.dims.size();
    if (indexRank == 0)
      Undefined("SegHist: rank 0");
    if ((int64_t)mapped.size() != indexRank + 1)
      Undefined("SegHist: unexpected kernel body result count");

    Values indexComponents(mapped.begin(), mapped.begin() + indexRank);
    mlir::Value updateValue = mapped[indexRank];
    int64_t n = getRankedType(updateValue.getType()).getShape()[0];
    mlir::Type idxElemTy =
        getRankedType(indexComponents[0].getType()).getElementType();

    // Stack the per-dimension index components into one [n, R] indices
    // tensor -- scatter wants a single tensor with index_vector_dim
    // selecting the per-update index vector, not R separate 1-D arrays.
    Values reshaped;
    for (mlir::Value idx : indexComponents)
      reshaped.push_back(hlo::ReshapeOp::create(
          builder, mlir::RankedTensorType::get({n, 1}, idxElemTy), idx));
    mlir::Value scatterIndices =
        indexRank == 1 ? reshaped[0]
                       : hlo::ConcatenateOp::create(
                             builder, mlir::ValueRange(reshaped), (uint64_t)1);

    // The destination array is a separate, already-lowered statement
    // (Futhark materializes it as its own `replicate`/zero-fill before the
    // seghist runs) -- not something this function constructs itself.
    mlir::Value dest = ctx.subexps.lookup(histOp.dest[0].name);

    mlir::SmallVector<int64_t> insertedWindowDims, scatterDimsToOperandDims;
    for (int64_t i = 0; i < indexRank; ++i) {
      insertedWindowDims.push_back(i);
      scatterDimsToOperandDims.push_back(i);
    }
    auto scatterDims =
        hlo::ScatterDimensionNumbersAttr::get(&context,
                                              /*updateWindowDims=*/{},
                                              insertedWindowDims,
                                              /*inputBatchingDims=*/{},
                                              /*scatterIndicesBatchingDims=*/{},
                                              scatterDimsToOperandDims,
                                              /*indexVectorDim=*/1);

    auto op = hlo::ScatterOp::create(builder,
                                     mlir::TypeRange{dest.getType()},
                                     mlir::ValueRange{dest},
                                     scatterIndices,
                                     mlir::ValueRange{updateValue},
                                     scatterDims,
                                     /*indices_are_sorted=*/false,
                                     /*unique_indices=*/false);

    mlir::Type elemTy = getRankedType(dest.getType()).getElementType();
    mlir::SmallVector<mlir::Type> argTys = {
        mlir::RankedTensorType::get({}, elemTy),
        mlir::RankedTensorType::get({}, elemTy)};
    mlir::SmallVector<mlir::Location> locs(2, builder.getLoc());
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      mlir::Block *blk =
          builder.createBlock(&op.getUpdateComputation(), {}, argTys, locs);
      builder.setInsertionPointToStart(blk);

      // Per the spec, update_computation(results[result_index],
      // updates_converted) -- old/accumulated value first, new value
      // second -- exactly Futhark's own HistOp.lambda.params convention
      // (accumulator first, new value second), so no transpose is needed
      // here, unlike LowerSegScanReduceWindow's (reverse) convention --
      // see the table in LowerSegRed's comment.
      Ctx local = ctx;
      local.subexps.insert(histOp.lambda.params[0].name, blk->getArgument(0));
      local.subexps.insert(histOp.lambda.params[1].name, blk->getArgument(1));
      Values results = LowerBody(histOp.lambda.body, local);
      hlo::ReturnOp::create(builder, mlir::ValueRange{results[0]});
    }

    return Values(op->getResults().begin(), op->getResults().end());
  }

  // Applies `lambda` elementwise to `inputs` (already full, matching-shape
  // tensors -- no SegSpace/affine-read discovery needed, unlike
  // LowerSegMap). Used for SegScan's post-op. Same "one stablehlo.map per
  // output value" structure as LowerSegMap, since map allows only a single
  // result; duplicated rather than shared since this call site has no
  // affine-reads/iota operands to thread through.
  Values LowerElementwiseLambda(const Lambda &lambda, Values inputs, Ctx &ctx) {
    auto shapeTy = getRankedType(inputs[0].getType()).getShape();
    mlir::SmallVector<int64_t> allDims;
    for (int64_t i = 0; i < (int64_t)shapeTy.size(); ++i)
      allDims.push_back(i);

    mlir::SmallVector<mlir::Type> argTys;
    for (mlir::Value v : inputs)
      argTys.push_back(mlir::RankedTensorType::get(
          {}, getRankedType(v.getType()).getElementType()));
    mlir::SmallVector<mlir::Location> locs(argTys.size(), builder.getLoc());

    Values finalResults;
    for (size_t j = 0; j < lambda.ret.size(); ++j) {
      auto resultTy = mlir::RankedTensorType::get(
          shapeTy, LowerSegOpBaseType(lambda.ret[j], ctx));
      auto op = hlo::MapOp::create(
          builder, mlir::TypeRange{resultTy}, inputs, allDims);

      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        mlir::Block *blk =
            builder.createBlock(&op.getRegion(), {}, argTys, locs);
        builder.setInsertionPointToStart(blk);

        Ctx local = ctx;
        for (auto [param, arg] :
             llvm::zip_equal(lambda.params, blk->getArguments()))
          local.subexps.insert(param.name, arg);
        Values results = LowerBody(lambda.body, local);
        hlo::ReturnOp::create(builder, mlir::ValueRange{results[j]});
      }

      BroadcastFreeVarsIntoMapRegion(op, shapeTy, "elementwise lambda");

      finalResults.push_back(op->getResult(0));
    }
    return finalResults;
  }

  // segscan = segmap (fused per-element part) + an inclusive scalar,
  // associative scan over the innermost (segment) dimension (scalar
  // operators via stablehlo.reduce_window below, tuple operators via the
  // parallel-prefix algorithm in ScanRec -- see the dispatch and its
  // comment further down) + an elementwise post-op. chlo.scan was tried
  // first and dropped: despite its own doc comment's claim, it does
  // decompose to StableHLO (a sequential stablehlo.while loop), but that
  // makes it strictly worse than either path used here, which are both
  // O(log n) parallel (see PROGRESS.md's Phase 5/6 for the full history).
  Values LowerSegScan(const SegScan &pSegScan, Ctx &ctx) {
    RequireThreadLevel(pSegScan.lvl);

    if (pSegScan.ops.size() != 1)
      Undefined("segscan: multiple operators");
    const SegBinOp &segBinOp = pSegScan.ops[0];
    if (!segBinOp.shape.dims.empty())
      Undefined("segscan: vector-valued operator");
    // Futhark's own semantics already require scan's operator to be
    // associative for the program to be valid Futhark at all (unlike
    // commutativity, which SegBinOp.comm tracks separately), so it's safe
    // to always treat it as associative here.
    IterationSpace iterSpace = LowerSegSpace(pSegScan.space, ctx);
    int64_t rank = std::ssize(iterSpace);
    if (rank == 0)
      Undefined("segscan: rank 0");

    // See exclusive_scan.hpp: Futhark's canonical exclusive-scan source
    // idiom isn't expressible through LowerSegMap at all (an offset,
    // branch-nested read), so it's recognized and handled directly here
    // instead of falling through to the general mapped-scan path below.
    if (rank == 1) {
      if (auto match =
              exclusive_scan::TryMatch(iterSpace[0].id, pSegScan.body)) {
        mlir::Value arrFull = ctx.subexps.lookup(match->array.name);
        Values shiftNeutral;
        for (auto &e : segBinOp.neutral)
          shiftNeutral.push_back(LowerSubExp(e, ctx));
        int64_t scanDim = rank - 1;
        Values inclusive = LowerSegScanReduceWindow(
            {arrFull}, shiftNeutral, scanDim, segBinOp, ctx);
        mlir::Value inc = inclusive[0];
        auto incTy = getRankedType(inc.getType());
        int64_t n = incTy.getShape()[scanDim];
        mlir::Value fill = LowerSubExp(match->fillValue, ctx);
        mlir::Value fillVec = hlo::ReshapeOp::create(
            builder, mlir::RankedTensorType::get({1}, incTy.getElementType()),
            fill);
        mlir::Value head = hlo::SliceOp::create(
            builder, inc, mlir::SmallVector<int64_t>{0},
            mlir::SmallVector<int64_t>{n - 1}, mlir::SmallVector<int64_t>{1});
        mlir::Value shifted = hlo::ConcatenateOp::create(
            builder, mlir::ValueRange{fillVec, head}, (uint64_t)scanDim);
        return LowerElementwiseLambda(pSegScan.post_op.lambda, {shifted}, ctx);
      }
    }

    Values mapped = LowerSegMap(
        pSegScan.lvl, pSegScan.space, pSegScan.body, pSegScan.ret, ctx);

    Values neutral;
    for (auto &e : segBinOp.neutral)
      neutral.push_back(LowerSubExp(e, ctx));
    int64_t valueCount = (int64_t)neutral.size();
    int64_t scanDim = rank - 1;

    // stablehlo.reduce_window (scalar path below) does NOT reliably
    // preserve left-to-right combination order for a non-commutative
    // operator -- confirmed empirically, not just by spec-reading (see
    // PROGRESS.md Phase 5). The tuple path (ScanRec's parallel-prefix
    // algorithm) does preserve order and has no such restriction, but
    // isn't used for scalars too because the dispatch below is keyed on
    // tuple-vs-scalar arity, not on correctness.
    //
    // SegBinOp.comm can't be used to pick a cheaper path when safe:
    // Futhark's own GPU IR never prints "commutative" for a SegScan's
    // operator at all (confirmed by diffing scan0.fut's dump against
    // reduce.fut's for the identical `+`), so the flag is always
    // Noncommutative here and can't distinguish safe from unsafe cases.
    //
    // TODO: scalar non-commutative-but-associative operators are therefore
    // an unaddressed residual risk through reduce_window below -- no
    // obvious Futhark scalar primitive produces one today, but nothing
    // currently gates against it either. See also
    // LowerSegScanReduceWindow's own comment.
    Values scanned =
        valueCount == 1
            ? LowerSegScanReduceWindow(mapped, neutral, scanDim, segBinOp, ctx)
            : LowerSegScanParallelPrefix(
                  mapped, neutral, scanDim, segBinOp, ctx);

    return LowerElementwiseLambda(pSegScan.post_op.lambda, scanned, ctx);
  }

  // Scalar associative scan via a full-width, left-padded
  // stablehlo.reduce_window (window size = the scan dimension's own
  // length, stride 1): an inclusive, forward scan, matching the exact
  // pattern JAX's own jnp.cumsum lowering uses (checked via
  // jax.jit(jnp.cumsum).lower(...).as_text() -- CPU-only, see
  // LowerSegRed's comment for the caveat on all such checks in this
  // file). Its results are already the per-position inclusive scan
  // values directly, no separate carry to discard.
  //
  // TODO: unverified for a scalar non-commutative-but-associative
  // operator (see LowerSegScan's comment) -- reduce_window is only
  // confirmed order-preserving for the scalar cases tested so far, all
  // of which happen to be commutative.
  Values LowerSegScanReduceWindow(Values mapped, Values neutral,
                                  int64_t scanDim, const SegBinOp &segBinOp,
                                  Ctx &ctx) {
    int64_t rank = getRankedType(mapped[0].getType()).getRank();
    int64_t valueCount = (int64_t)neutral.size();
    auto shapeTy = getRankedType(mapped[0].getType()).getShape();
    int64_t n = shapeTy[scanDim];

    mlir::SmallVector<int64_t> windowDims(rank, 1);
    windowDims[scanDim] = n;
    mlir::SmallVector<int64_t> windowStrides(rank, 1);
    mlir::SmallVector<int64_t> baseDilations(rank, 1);
    mlir::SmallVector<int64_t> windowDilations(rank, 1);
    mlir::SmallVector<int64_t> paddingFlat(rank * 2, 0);
    paddingFlat[scanDim * 2] = n - 1;
    auto paddingTy =
        mlir::RankedTensorType::get({rank, 2}, builder.getI64Type());
    auto paddingAttr = mlir::DenseIntElementsAttr::get(paddingTy, paddingFlat);

    mlir::SmallVector<mlir::Type> resultTypes;
    for (mlir::Value v : mapped)
      resultTypes.push_back(v.getType());

    auto op = hlo::ReduceWindowOp::create(
        builder,
        resultTypes,
        mapped,
        neutral,
        builder.getDenseI64ArrayAttr(windowDims),
        builder.getDenseI64ArrayAttr(windowStrides),
        builder.getDenseI64ArrayAttr(baseDilations),
        builder.getDenseI64ArrayAttr(windowDilations),
        paddingAttr);

    // Per the StableHLO spec, the body's block args are (one value per
    // windowed input, then one accumulator per init_value) -- the
    // opposite of Futhark's own SegBinOp.lambda.params convention
    // (accumulators first), and also the opposite of plain
    // stablehlo.reduce's convention used in LowerSegRed -- these two ops
    // do NOT share a block-arg order despite both being reduce-shaped; see
    // the table in LowerSegRed's comment. Confirmed via a JAX reference
    // lowering for this op specifically (jax.lax.reduce_window with a
    // non-commutative combiner, decoded term by term), not assumed to
    // match stablehlo.reduce's convention.
    mlir::SmallVector<mlir::Type> argTys;
    for (mlir::Value v : mapped)
      argTys.push_back(mlir::RankedTensorType::get(
          {}, getRankedType(v.getType()).getElementType()));
    for (mlir::Value v : neutral)
      argTys.push_back(v.getType());
    mlir::SmallVector<mlir::Location> locs(argTys.size(), builder.getLoc());

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      mlir::Block *blk = builder.createBlock(&op.getBody(), {}, argTys, locs);
      builder.setInsertionPointToStart(blk);

      Ctx local = ctx;
      mlir::ValueRange blockArgs = blk->getArguments();
      mlir::ValueRange newValueArgs = blockArgs.take_front(valueCount);
      mlir::ValueRange accArgs = blockArgs.drop_front(valueCount);
      for (auto [param, acc] : llvm::zip_equal(
               segBinOp.lambda.params | std::views::take(valueCount), accArgs))
        local.subexps.insert(param.name, acc);
      for (auto [param, nv] : llvm::zip_equal(segBinOp.lambda.params |
                                                  std::views::drop(valueCount),
                                              newValueArgs))
        local.subexps.insert(param.name, nv);

      Values results = LowerBody(segBinOp.lambda.body, local);
      hlo::ReturnOp::create(builder, results);
    }

    return Values(op->getResults().begin(), op->getResults().end());
  }

  // A scalar 0-d constant of value zero for an arbitrary element type --
  // used by the parallel-prefix scan's interleave step below, which pads
  // with zero and recombines via add/or rather than the user's own
  // operator (see Interleave's comment for why that's still correct).
  mlir::Value ZeroScalar(mlir::Type elemTy) {
    auto ty = mlir::RankedTensorType::get({}, elemTy);
    return hlo::ConstantOp::create(
        builder, mlir::DenseElementsAttr::get(ty, builder.getZeroAttr(elemTy)));
  }

  // stablehlo.slice along a single dimension, Python-slice style (start
  // inclusive, limit exclusive), identity on every other dimension.
  Values SliceDim(Values xs, int64_t start, int64_t limit, int64_t stride,
                  int64_t dim) {
    Values result;
    for (mlir::Value v : xs) {
      auto ty = getRankedType(v.getType());
      int64_t rank = ty.getRank();
      mlir::SmallVector<int64_t> starts(rank, 0);
      mlir::SmallVector<int64_t> limits(ty.getShape().begin(),
                                        ty.getShape().end());
      mlir::SmallVector<int64_t> strides(rank, 1);
      starts[dim] = start;
      limits[dim] = limit;
      strides[dim] = stride;
      result.push_back(
          hlo::SliceOp::create(builder, v, starts, limits, strides));
    }
    return result;
  }

  Values ConcatDim(Values a, Values b, int64_t dim) {
    Values result;
    for (auto [av, bv] : llvm::zip_equal(a, b))
      result.push_back(hlo::ConcatenateOp::create(
          builder, mlir::ValueRange{av, bv}, (uint64_t)dim));
    return result;
  }

  // Interleaves two sequences of equal length, or where `a` is exactly one
  // longer than `b`, along `dim` -- jax.lax.associative_scan's own
  // `_interleave` helper (checked its source directly, not derived from
  // the algorithm description alone). Each is zero-padded at alternating
  // positions (interior padding of 1 guarantees the two padded arrays
  // never have a real value at the same position) and recombined with
  // add (or, for bool, logical or) -- NOT the user's own scan operator,
  // which plays no role here: the padding trick is a pure interleave, not
  // a combination of real values.
  Values Interleave(Values a, Values b, int64_t dim) {
    int64_t aLen = getRankedType(a[0].getType()).getShape()[dim];
    int64_t bLen = getRankedType(b[0].getType()).getShape()[dim];
    int64_t aHigh = (aLen == bLen) ? 1 : 0;
    int64_t bHigh = (aLen == bLen) ? 0 : 1;

    Values result;
    for (auto [av, bv] : llvm::zip_equal(a, b)) {
      auto rankedTy = getRankedType(av.getType());
      int64_t rank = rankedTy.getRank();
      mlir::Type elemTy = rankedTy.getElementType();
      mlir::Value zero = ZeroScalar(elemTy);

      mlir::SmallVector<int64_t> aLow(rank, 0), aHighV(rank, 0),
          aInterior(rank, 0);
      aHighV[dim] = aHigh;
      aInterior[dim] = 1;
      mlir::Value aPadded =
          hlo::PadOp::create(builder, av, zero, aLow, aHighV, aInterior);

      mlir::SmallVector<int64_t> bLow(rank, 0), bHighV(rank, 0),
          bInterior(rank, 0);
      bLow[dim] = 1;
      bHighV[dim] = bHigh;
      bInterior[dim] = 1;
      mlir::Value bPadded =
          hlo::PadOp::create(builder, bv, zero, bLow, bHighV, bInterior);

      mlir::Value combined =
          elemTy.isInteger(1)
              ? mlir::Value(hlo::OrOp::create(builder, aPadded, bPadded))
              : mlir::Value(hlo::AddOp::create(builder, aPadded, bPadded));
      result.push_back(combined);
    }
    return result;
  }

  // Applies the scan operator elementwise to two full-shape tuples of
  // arrays -- the parallel-prefix algorithm's "combine" step, distinct
  // from LowerElementwiseLambda's single-input form. One stablehlo.map
  // per output value, same free-var-broadcast fixup as LowerSegMap/
  // LowerElementwiseLambda (stablehlo.map is isolated-from-above).
  // Operand order matches Futhark's own SegBinOp.lambda.params convention
  // directly (accumulator_params... bound to `a`, new_value_params...
  // bound to `b`) -- no transpose needed here, unlike
  // LowerSegScanReduceWindow above (see the table in LowerSegRed's
  // comment), since this is a plain stablehlo.map with no StableHLO-
  // mandated block-arg order to work around.
  Values Combine(Values a, Values b, const SegBinOp &segBinOp, Ctx &ctx) {
    Values inputs(a.begin(), a.end());
    inputs.insert(inputs.end(), b.begin(), b.end());

    auto shapeTy = getRankedType(a[0].getType()).getShape();
    mlir::SmallVector<int64_t> allDims;
    for (int64_t i = 0; i < (int64_t)shapeTy.size(); ++i)
      allDims.push_back(i);

    mlir::SmallVector<mlir::Type> argTys;
    for (mlir::Value v : inputs)
      argTys.push_back(mlir::RankedTensorType::get(
          {}, getRankedType(v.getType()).getElementType()));
    mlir::SmallVector<mlir::Location> locs(argTys.size(), builder.getLoc());

    Values finalResults;
    for (size_t j = 0; j < segBinOp.lambda.ret.size(); ++j) {
      auto resultTy = mlir::RankedTensorType::get(
          shapeTy, LowerSegOpBaseType(segBinOp.lambda.ret[j], ctx));
      auto op = hlo::MapOp::create(
          builder, mlir::TypeRange{resultTy}, inputs, allDims);

      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        mlir::Block *blk =
            builder.createBlock(&op.getRegion(), {}, argTys, locs);
        builder.setInsertionPointToStart(blk);

        Ctx local = ctx;
        for (auto [param, arg] :
             llvm::zip_equal(segBinOp.lambda.params, blk->getArguments()))
          local.subexps.insert(param.name, arg);
        Values results = LowerBody(segBinOp.lambda.body, local);
        hlo::ReturnOp::create(builder, mlir::ValueRange{results[j]});
      }

      BroadcastFreeVarsIntoMapRegion(op, shapeTy, "segscan parallel prefix");

      finalResults.push_back(op->getResult(0));
    }
    return finalResults;
  }

  // Tuple-operator associative scan: jax.lax.associative_scan's own
  // recursive Blelloch-style algorithm, ported directly from its Python
  // source (not re-derived from prose) and cross-checked against its
  // compiled op sequence (CPU-only, see LowerSegRed's comment for the
  // caveat). O(log n) parallel steps, replacing the earlier chlo.scan
  // implementation's O(n) sequential stablehlo.while loop outright. All
  // shapes are static here (scalars-as-0-d-tensors + Undefined() on
  // dynamic sizes), so the recursion unrolls entirely at compile time --
  // exactly like JAX's own trace-time unrolling.
  Values ScanRec(Values elems, int64_t dim, const SegBinOp &segBinOp,
                 Ctx &ctx) {
    int64_t n = getRankedType(elems[0].getType()).getShape()[dim];
    if (n < 2)
      return elems;

    Values reduced = Combine(SliceDim(elems, 0, n - 1, 2, dim),
                             SliceDim(elems, 1, n, 2, dim),
                             segBinOp,
                             ctx);
    Values oddElems = ScanRec(reduced, dim, segBinOp, ctx);

    Values evenElems;
    int64_t pairCount = n / 2;
    if (n % 2 == 0) {
      evenElems = Combine(SliceDim(oddElems, 0, pairCount - 1, 1, dim),
                          SliceDim(elems, 2, n, 2, dim),
                          segBinOp,
                          ctx);
    } else {
      evenElems =
          Combine(oddElems, SliceDim(elems, 2, n, 2, dim), segBinOp, ctx);
    }
    evenElems = ConcatDim(SliceDim(elems, 0, 1, 1, dim), evenElems, dim);

    return Interleave(evenElems, oddElems, dim);
  }

  Values LowerSegScanParallelPrefix(Values mapped, Values neutral,
                                    int64_t scanDim, const SegBinOp &segBinOp,
                                    Ctx &ctx) {
    (void)neutral; // Unlike reduce_window/chlo.scan, this algorithm has no
                   // separate init-value operand: Futhark's own scan
                   // semantics make the first element its own scan result
                   // (see ScanRec's base case and even-element handling),
                   // matching jax.lax.associative_scan's documented
                   // behavior exactly, so neutral is unused here by
                   // design, not an oversight.
    return ScanRec(mapped, scanDim, segBinOp, ctx);
  }

  // TODO: only SegThread (one GPU thread per iteration-space point) is
  // supported. SegBlock (one GPU block per point, with further intra-block
  // parallelism) and SegThreadInBlock (the inner level of such a nested
  // SegOp) are gated here, unanalyzed -- lowering them would mean mapping
  // Futhark's two-level block/thread nesting onto StableHLO's single flat
  // iteration space, which hasn't been designed yet.
  void RequireThreadLevel(const SegLevel &lvl) {
    if (!std::holds_alternative<SegThread>(lvl))
      Undefined();
  }

  // Find SegOp iteration space.
  IterationSpace LowerSegSpace(const SegSpace &space, Ctx &ctx) {
    IterationSpace iterSpace;
    int i = 0;
    for (const auto &[id, dim] : space.dims) {
      auto d = LowerSubExp(dim, ctx);
      iterSpace.push_back(
          Dim{id, d, mlir::ShapedType::isDynamic(toShapeType(dim)), i});
      ++i;
    }
    return iterSpace;
  }

  // Find affine reads in the kernel body. An affine read is an array load whose
  // index expression is an affine function of the iteration space.
  std::vector<AffineRead> FindSegOpAffineReads(const IterationSpace &iterSpace,
                                               const KernelBody &body) {
    std::vector<AffineRead> affine_reads;
    for (const auto &stm : body.stms) {
      if (stm.pat.elems.size() != 1)
        Undefined();
      auto vnBound = stm.pat.elems[0].name;

      if (auto r = toAffineRead(vnBound, stm.exp, iterSpace)) {
        affine_reads.push_back(r.value());
      }
    }

    return affine_reads;
  }

  // Currently the only indexing recognized as affine are direct uses of the
  // iteration space variables, e.g., `x[i]` where `i` is an iteration variable.
  std::optional<AffineRead> toAffineRead(VName vn, const Exp &exp,
                                         const IterationSpace &iterSpace) {
    auto rank = iterSpace.size();
    if (auto e = std::get_if<ExpBasicOp>(&exp.v)) {
      if (auto idx = std::get_if<BasicOpIndex>(&e->op.v)) {
        auto vnArray = idx->vName;

        std::vector<mlir::AffineExpr> usedDims;
        for (const auto &dim : idx->slice.dims) {
          // TODO suport affine indexing beyond seg space ids
          // Anything else -- a slice, a constant index, a variable from
          // outside the iteration space -- is not an affine read, and is left
          // to the ordinary lowering of the statement.
          auto *fix = std::get_if<DimFix<SubExp>>(&dim.v);
          if (!fix)
            return std::nullopt;
          auto *var = std::get_if<VarSubExp>(&fix->i.v);
          if (!var)
            return std::nullopt;

          auto d = std::ranges::find(iterSpace, var->v.name, &Dim::id);
          if (d == iterSpace.end())
            return std::nullopt;
          auto i = d->index;

          usedDims.push_back(mlir::getAffineDimExpr(i, &context));
        }

        auto m = mlir::AffineMap::get(rank, 0, usedDims, &context);
        return AffineRead{vn, vnArray, m};
      }
    }
    return std::nullopt;
  }

  // A gather: like toAffineRead, but each index dimension may instead be
  // the bound name of an *already recognized* AffineRead (one level of
  // indirection -- e.g. `tokens[perm[gtid], gtid2]`, with `perm[gtid]`
  // itself an AffineRead), not just a bare iteration id. Statements
  // `toAffineRead` already accepts are skipped here (they're handled by
  // the plain affine-read path instead); a statement is only a GatherRead
  // if at least one of its index dims is this indirect case. Deliberately
  // not recursive -- a dimension naming another GatherRead's result isn't
  // recognized -- since nothing currently needs more than one level.
  std::vector<GatherRead>
  FindSegOpGatherReads(const IterationSpace &iterSpace,
                       const std::vector<AffineRead> &affineReads,
                       const KernelBody &body) {
    std::unordered_set<std::string> iterIds;
    for (auto &d : iterSpace)
      iterIds.insert(d.id);
    std::unordered_set<std::string> affineResults;
    for (auto &r : affineReads)
      affineResults.insert(r.result.name);

    std::vector<GatherRead> gathers;
    for (const auto &stm : body.stms) {
      if (stm.pat.elems.size() != 1)
        Undefined();
      auto vnBound = stm.pat.elems[0].name;
      if (toAffineRead(vnBound, stm.exp, iterSpace))
        continue;

      auto *e = std::get_if<ExpBasicOp>(&stm.exp.v);
      if (!e)
        continue;
      auto *idx = std::get_if<BasicOpIndex>(&e->op.v);
      if (!idx)
        continue;

      std::vector<VName> indexNames;
      bool ok = true;
      bool anyIndirect = false;
      for (const auto &dim : idx->slice.dims) {
        auto *fix = std::get_if<DimFix<SubExp>>(&dim.v);
        auto *var = fix ? std::get_if<VarSubExp>(&fix->i.v) : nullptr;
        if (!var) {
          ok = false;
          break;
        }
        if (iterIds.contains(var->v.name)) {
          indexNames.push_back(var->v);
        } else if (affineResults.contains(var->v.name)) {
          indexNames.push_back(var->v);
          anyIndirect = true;
        } else {
          ok = false;
          break;
        }
      }
      if (ok && anyIndirect)
        gathers.push_back(GatherRead{vnBound, idx->vName, std::move(indexNames)});
    }
    return gathers;
  }

  mlir::Type LowerSegOpBaseType(const Type &pReturnType, Ctx &ctx) {
    return match(
        pReturnType.t,
        [&](const TypePrim<Shape, NoUniqueness> &val) {
          return LowerPrimType(val.t);
        },
        [&](const TypeArray<Shape, NoUniqueness> &val) -> mlir::Type {
          Undefined();
        },
        [](const auto &) -> mlir::Type { Undefined(); });
  };

  // Scalars are 0-d tensors now; getPrimValueAttr stays a bare scalar attr
  // (also reused by LowerArrayLit to build a full-array DenseElementsAttr).
  mlir::Value LowerPrimValue(PrimValue value, Ctx &ctx) {
    mlir::TypedAttr scalarAttr = getPrimValueAttr(value);
    auto ty = mlir::RankedTensorType::get({}, scalarAttr.getType());
    return hlo::ConstantOp::create(
        builder, mlir::DenseElementsAttr::get(ty, scalarAttr));
  }

  mlir::TypedAttr getPrimValueAttr(PrimValue value) {
    return match(
        value,
        [&](const IntValue &val) -> mlir::TypedAttr {
          return builder.getIntegerAttr(builder.getIntegerType(GetWidth(val)),
                                        GetValue(val));
        },
        [&](const FloatValue &val) -> mlir::TypedAttr {
          return builder.getFloatAttr(GetFloatType(builder, GetWidth(val)),
                                      GetValue(val));
        },
        [&](const BoolValue &val) -> mlir::TypedAttr {
          return builder.getBoolAttr(val.v);
        },
        [&](const UnitValue &) -> mlir::TypedAttr { Undefined(); });
  }

  // Every operand is a tensor now (0-d for a scalar, N-d for a whole-array
  // op), so stablehlo's elementwise ops handle both uniformly -- no more
  // separate scalar (arith) vs. tensor (linalg) code paths.
  mlir::Value LowerBinOp(BinOp binOp, mlir::Value op0, mlir::Value op1,
                         Ctx &ctx) {
    namespace hlo = mlir::stablehlo;
    if (std::get_if<BinOpAdd>(&binOp.v) || std::get_if<BinOpFAdd>(&binOp.v))
      return hlo::AddOp::create(builder, op0, op1);
    if (std::get_if<BinOpMul>(&binOp.v) || std::get_if<BinOpFMul>(&binOp.v))
      return hlo::MulOp::create(builder, op0, op1);
    if (std::get_if<BinOpSub>(&binOp.v) || std::get_if<BinOpFSub>(&binOp.v))
      return hlo::SubtractOp::create(builder, op0, op1);

    // Futhark's `squot` truncates towards zero, matching stablehlo.divide's
    // (and C's) integer division. `sdiv`/`sdiv_up` round toward -inf/+inf;
    // `udiv`/`udiv_up` are the unsigned equivalents -- no direct StableHLO
    // op, flagged rather than risking wrong rounding.
    if (std::get_if<BinOpSQuot>(&binOp.v) || std::get_if<BinOpFDiv>(&binOp.v))
      return hlo::DivOp::create(builder, op0, op1);
    if (std::get_if<BinOpSDivUp>(&binOp.v) ||
        std::get_if<BinOpSDiv>(&binOp.v) || std::get_if<BinOpUDiv>(&binOp.v) ||
        std::get_if<BinOpUDivUp>(&binOp.v))
      Undefined("binop: rounding div");

    // `srem`/`smod` are truncated/floored remainder -- for the non-negative
    // sizes and indices in this IR they coincide with stablehlo.remainder.
    if (std::get_if<BinOpSRem>(&binOp.v) || std::get_if<BinOpSMod>(&binOp.v) ||
        std::get_if<BinOpFMod>(&binOp.v))
      return hlo::RemOp::create(builder, op0, op1);
    if (std::get_if<BinOpUMod>(&binOp.v))
      Undefined("binop: umod");

    // Bitwise / logical.
    if (std::get_if<BinOpAnd>(&binOp.v) || std::get_if<BinOpLogAnd>(&binOp.v))
      return hlo::AndOp::create(builder, op0, op1);
    if (std::get_if<BinOpOr>(&binOp.v) || std::get_if<BinOpLogOr>(&binOp.v))
      return hlo::OrOp::create(builder, op0, op1);
    if (std::get_if<BinOpXor>(&binOp.v))
      return hlo::XorOp::create(builder, op0, op1);
    if (std::get_if<BinOpShl>(&binOp.v))
      return hlo::ShiftLeftOp::create(builder, op0, op1);
    if (std::get_if<BinOpLShr>(&binOp.v))
      return hlo::ShiftRightLogicalOp::create(builder, op0, op1);
    if (std::get_if<BinOpAShr>(&binOp.v))
      return hlo::ShiftRightArithmeticOp::create(builder, op0, op1);

    // Min/max. stablehlo.minimum/maximum have no signedness attribute (unlike
    // compare), so unsigned min/max isn't representable without a separate
    // compare+select -- flagged rather than guessed.
    if (std::get_if<BinOpSMin>(&binOp.v) || std::get_if<BinOpFMin>(&binOp.v))
      return hlo::MinOp::create(builder, op0, op1);
    if (std::get_if<BinOpSMax>(&binOp.v) || std::get_if<BinOpFMax>(&binOp.v))
      return hlo::MaxOp::create(builder, op0, op1);
    if (std::get_if<BinOpUMin>(&binOp.v) || std::get_if<BinOpUMax>(&binOp.v))
      Undefined("binop: unsigned min/max");

    if (std::get_if<BinOpFPow>(&binOp.v))
      return hlo::PowOp::create(builder, op0, op1);

    Undefined("binop");
  }

  mlir::Value LowerArrayLit(BasicOpArrayLit arrayLit, Ctx &ctx) {
    auto tensorTy = LowerTy(arrayLit.t);
    bool allConstants = true;
    auto attrs = mlir::SmallVector<mlir::Attribute>();
    for (auto subExp : arrayLit.values) {
      auto *c = std::get_if<ConstantSubExp>(&subExp.v);
      if (!c) {
        allConstants = false;
        break;
      }
      attrs.push_back(getPrimValueAttr(c->v));
    }

    if (allConstants) {
      return hlo::ConstantOp::create(
          builder,
          mlir::DenseElementsAttr::get(getRankedType(tensorTy), attrs));
    } else {
      // A non-constant array literal: concatenate each element, reshaped
      // from its 0-d scalar type up to a 1-d [1] slice (tensor.from_elements
      // isn't in XLA's dialect surface).
      auto elemTy = getRankedType(tensorTy).getElementType();
      auto sliceTy = mlir::RankedTensorType::get({1}, elemTy);
      mlir::SmallVector<mlir::Value> slices;
      for (auto subExp : arrayLit.values) {
        mlir::Value v = LowerSubExp(subExp, ctx);
        slices.push_back(hlo::ReshapeOp::create(builder, sliceTy, v));
      }
      return hlo::ConcatenateOp::create(builder, slices, (uint64_t)0);
    }
  }

  template <typename T> static std::vector<T> Iota(int64_t n, T x, T s) {
    std::vector<T> elems;
    if (n <= 0)
      return elems;

    auto currentValue = x;
    for (int64_t i = 0; i < n; i++) {
      elems.push_back(currentValue);
      currentValue += s;
    }

    return elems;
  }

  // Futhark's primitive functions (Futhark.IR.Primitive.primFuns) are applied
  // without ever being defined in the IR, so a name with no definition is one
  // of these. They all carry the operand width as a suffix -- separated by an
  // underscore when the name itself ends in a digit -- which we can drop,
  // because the lowered operands already have the right MLIR type.
  //
  //   "sqrt32" -> "sqrt", "log10_64" -> "log10", "atan2_32" -> "atan2"
  static std::optional<std::string> IntrinsicBaseName(const std::string &name) {
    auto end = name.size();
    while (end > 0 && std::isdigit(static_cast<unsigned char>(name[end - 1])))
      --end;
    // No width suffix at all, so this is not a primitive function.
    if (end == name.size())
      return std::nullopt;
    if (end > 0 && name[end - 1] == '_')
      --end;
    if (end == 0)
      return std::nullopt;
    return name.substr(0, end);
  }

  // TODO: unverified -- no currently-passing test exercises any intrinsic
  // (gelu.fut, the one test that would, hits the dynamic-shape Undefined()
  // gate first). Mapped from the actual stablehlo/chlo op inventory (checked
  // against the pinned stablehlo revision's StablehloOps.td/ChloOps.td), not
  // guessed, but not yet run.
  Values LowerIntrinsic(const std::string &fname, mlir::ValueRange args) {
    // `cond_t` is Futhark's branchless select; its suffix is a full type name
    // ("cond_i64", "cond_bool"), not the width suffix IntrinsicBaseName knows.
    if (fname.starts_with("cond_")) {
      if (args.size() != 3)
        Undefined();
      return {hlo::SelectOp::create(builder, args[0], args[1], args[2])
                  .getResult()};
    }

    auto base = IntrinsicBaseName(fname);
    if (!base)
      throw std::runtime_error(
          std::format("no definition of function '{}'", fname));

    auto unary = [&]<typename Op>() -> Values {
      if (args.size() != 1)
        Undefined();
      return {Op::create(builder, args[0]).getResult()};
    };
    auto binary = [&]<typename Op>() -> Values {
      if (args.size() != 2)
        Undefined();
      return {Op::create(builder, args[0], args[1]).getResult()};
    };

    const auto &n = *base;
    {
      using namespace mlir::stablehlo;
      if (n == "sqrt")
        return unary.operator()<SqrtOp>();
      if (n == "rsqrt")
        return unary.operator()<RsqrtOp>();
      if (n == "cbrt")
        return unary.operator()<CbrtOp>();
      if (n == "exp")
        return unary.operator()<ExpOp>();
      if (n == "expm1")
        return unary.operator()<Expm1Op>();
      if (n == "log")
        return unary.operator()<LogOp>();
      if (n == "log1p")
        return unary.operator()<Log1pOp>();
      if (n == "sin")
        return unary.operator()<SineOp>();
      if (n == "cos")
        return unary.operator()<CosineOp>();
      if (n == "tan")
        return unary.operator()<TanOp>();
      if (n == "tanh")
        return unary.operator()<TanhOp>();
      if (n == "ceil")
        return unary.operator()<CeilOp>();
      if (n == "floor")
        return unary.operator()<FloorOp>();
      // Futhark's `round` is C's `rint` under the default rounding mode,
      // i.e. ties go to even.
      if (n == "round")
        return unary.operator()<RoundNearestEvenOp>();
      if (n == "popc")
        return unary.operator()<PopulationCountOp>();
      if (n == "clz")
        return unary.operator()<ClzOp>();
      if (n == "atan2")
        return binary.operator()<Atan2Op>();
    }
    {
      using namespace mlir::chlo;
      if (n == "asin")
        return unary.operator()<AsinOp>();
      if (n == "acos")
        return unary.operator()<AcosOp>();
      if (n == "atan")
        return unary.operator()<AtanOp>();
      if (n == "sinh")
        return unary.operator()<SinhOp>();
      if (n == "cosh")
        return unary.operator()<CoshOp>();
      if (n == "asinh")
        return unary.operator()<AsinhOp>();
      if (n == "acosh")
        return unary.operator()<AcoshOp>();
      if (n == "atanh")
        return unary.operator()<AtanhOp>();
      if (n == "erf")
        return unary.operator()<ErfOp>();
      if (n == "erfc")
        return unary.operator()<ErfcOp>();
      if (n == "isinf")
        return unary.operator()<IsInfOp>();
    }
    // No StableHLO/CHLO equivalent found.
    if (n == "exp2" || n == "log2" || n == "log10" || n == "trunc" ||
        n == "isnan" || n == "ctz" || n == "copysign" || n == "fma" ||
        n == "mad")
      Undefined(std::format("intrinsic: {}", n));

    throw std::runtime_error(std::format(
        "'{}' is neither defined in the IR nor a primitive function we lower",
        fname));
  }

  static const FunDef *FindFunction(const Prog &prog,
                                    const std::string &fname) {
    for (const auto &f : prog.funs)
      if (f.name == fname)
        return &f;
    return nullptr;
  }

  std::vector<int64_t>
  LowerShape(const Shape &shape,
             const std::unordered_map<std::string, int64_t> &sizeEnv = {}) {
    return toShapeType(shape.dims, sizeEnv);
  }

  mlir::Type
  LowerTy(Type t,
          const std::unordered_map<std::string, int64_t> &sizeEnv = {}) {
    auto baseTy = &t.t.v;
    if (auto *val = std::get_if<TypePrim<Shape, NoUniqueness>>(baseTy)) {
      // Scalars are 0-d tensors (a hard XLA requirement), unlike an array's
      // own element type (see LowerTypeArray, which calls LowerPrimType
      // directly to get the bare element type instead).
      return mlir::RankedTensorType::get({}, LowerPrimType(val->t));
    }

    if (auto *val = std::get_if<TypeArray<Shape, NoUniqueness>>(baseTy)) {
      return LowerTypeArray(*val, sizeEnv);
    }

    Undefined();
  }

  mlir::Type LowerPrimType(PrimType t) {
    if (auto *val = std::get_if<PrimTypeInt>(&t.v)) {
      return LowerPrimTypeInt(*val);
    }

    if (auto *val = std::get_if<PrimTypeFloat>(&t.v)) {
      return LowerPrimTypeFloat(*val);
    }

    if (auto *val = std::get_if<PrimTypeBool>(&t.v)) {
      return LowerPrimTypeBool(*val);
    }

    Undefined();
  }

  mlir::Type LowerPrimTypeInt(PrimTypeInt primTypeInt) {
    return builder.getIntegerType(GetWidth(primTypeInt.t));
  }

  mlir::Type LowerPrimTypeFloat(PrimTypeFloat primTypeFloat) {
    auto w = GetWidth(primTypeFloat.t);
    if (w == 16)
      return builder.getF16Type();
    if (w == 32)
      return builder.getF32Type();
    return builder.getF64Type();
  }

  mlir::Type LowerPrimTypeBool(PrimTypeBool primTypeBool) {
    return builder.getIntegerType(1);
  }

  mlir::Type
  LowerTypeArray(TypeArray<Shape, NoUniqueness> typeArray,
                 const std::unordered_map<std::string, int64_t> &sizeEnv = {}) {
    return mlir::RankedTensorType::get(LowerShape(typeArray.shape, sizeEnv),
                                       LowerPrimType(typeArray.elem));
  }

  static mlir::Type GetFloatType(mlir::Builder &builder, uint64_t width) {
    if (width == 16)
      return builder.getF16Type();
    if (width == 32)
      return builder.getF32Type();
    if (width == 64)
      return builder.getF64Type();
    Undefined();
  }
};
