#pragma once
// Represents the state of a Futhark function
#include "blackbox.hpp"
#include "debug.hpp"
#include "ir/core.hpp"
#include "ir/segop.hpp"
#include "ir/syntax.hpp"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "utils.hpp"
#include "llvm/Support/ErrorHandling.h"
#include <cctype>
#include <format>
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

inline int64_t toShapeType(SubExp dim) {
  return match(
      dim,
      [&](const ConstantSubExp &c) { return c.GetIntValue(); },
      [&](const VarSubExp &) { return mlir::ShapedType::kDynamic; });
}

inline bool IsUnitType(const Type &type) {
  auto *prim = std::get_if<TypePrim<Shape, NoUniqueness>>(&type.t.v);
  return prim && std::holds_alternative<PrimTypeUnit>(prim->t.v);
}

inline const TypeAcc<Shape, NoUniqueness> *GetAccType(const Type &type) {
  return std::get_if<TypeAcc<Shape, NoUniqueness>>(&type.t.v);
}

inline bool IsAccType(const Type &type) {
  return GetAccType(type) != nullptr;
}

template <typename R, typename V>
concept Iterable = std::ranges::input_range<R> &&
                   std::same_as<std::ranges::range_value_t<R>, V>;

template <Iterable<SubExp> R>
inline std::vector<int64_t> toShapeType(const R &dims) {
  std::vector<int64_t> dimsTy;
  for (auto &d : dims)
    dimsTy.push_back(toShapeType(d));
  return dimsTy;
}

struct FutharkCompiler {
  using Values = mlir::SmallVector<mlir::Value>;

  Prog &prog;

  std::unordered_map<FunDef, mlir::func::FuncOp, FunDefHasher> functions;

  mlir::MLIRContext &context;

  mlir::ImplicitLocOpBuilder &builder;

  mlir::ModuleOp module;

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
    context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
    context.getOrLoadDialect<mlir::chlo::ChloDialect>();
    context.getOrLoadDialect<mlir::vector::VectorDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
  }

  std::variant<mlir::func::FuncOp, BlackBox> LowerFunction(const FunDef &fun) {
    if (auto blackbox = MaybeBlackBox(fun.attrs))
      return blackbox.value();

    if (functions.find(fun) != functions.end())
      return functions[fun];

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());

    std::vector<mlir::Type> retTypes;
    for (const auto &t : fun.retType) {
      retTypes.push_back(LowerTy(t.first.v));
    }
    // TODO: Maybe this is not a way to go
    auto runtimeParams = fun.params | std::views::filter([](const auto &param) {
                           return !IsUnitType(param.dec.v);
                         });
    std::vector<mlir::Type> inputTypes;
    for (const auto &param : runtimeParams) {
      inputTypes.push_back(LowerTy(param.dec.v));
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
          auto returns = match(
              LowerFunction(fun),
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
          // TODO(StableHLO rewrite, Phase 4): still scf.if (banned by XLA);
          // needs stablehlo.if. The condition is a 0-d i1 tensor now (scalars
          // are 0-d tensors), so it's unwrapped here for scf.if's bare i1.
          mlir::Value cond = mlir::tensor::ExtractOp::create(
              builder, LowerSubExp(e.cond, ctx), mlir::ValueRange{});
          return mlir::scf::IfOp::create(
                     builder,
                     cond,
                     [&](mlir::OpBuilder &b, mlir::Location loc) {
                       mlir::OpBuilder::InsertionGuard guard(builder);
                       builder.setInsertionPointToEnd(b.getInsertionBlock());
                       auto thenValue = LowerBody(*e.then_body, ctx);
                       mlir::scf::YieldOp::create(builder, loc, thenValue);
                     },
                     [&](mlir::OpBuilder &b, mlir::Location loc) {
                       mlir::OpBuilder::InsertionGuard guard(builder);
                       builder.setInsertionPointToEnd(b.getInsertionBlock());
                       auto elseValue = LowerBody(*e.else_body, ctx);
                       mlir::scf::YieldOp::create(builder, loc, elseValue);
                     })
              .getResults();
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
      return mlir::stablehlo::ReshapeOp::create(builder, tTy, value);
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
  // below. Precision defaults to DEFAULT; example_run.py's existing
  // --precision/set_precision text rewrite handles ad hoc tuning instead of
  // threading a compiler-level flag for it.
  Values LowerMatMul(mlir::ValueRange args, mlir::TypeRange retTypes,
                     Ctx &ctx) {
    assert(args.size() == 5);
    assert(retTypes.size() == 1);
    auto x = args[3];
    auto y = args[4];
    auto resultTy = getRankedType(retTypes[0]);
    if (!resultTy.hasStaticShape())
      Undefined("matmul: dynamic shape");

    auto dotDims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        &context, /*lhsBatchingDimensions=*/{}, /*rhsBatchingDimensions=*/{},
        /*lhsContractingDimensions=*/{1}, /*rhsContractingDimensions=*/{0});
    auto op = mlir::stablehlo::DotGeneralOp::create(builder, resultTy, x, y,
                                                     dotDims, {}, {});
    return op->getResults();
  }

  Values LowerUpdateAcc(const BasicOpUpdateAcc &update, Ctx &ctx) {
    assert(update.indices.size() == 1);
    assert(update.values.size() == 1);
    mlir::Value index = LowerSubExp(update.indices[0], ctx);
    mlir::Value value = LowerSubExp(update.values[0], ctx);
    return {value, index};
  }

  // TODO(StableHLO rewrite): with_acc scattering used to go through IREE's
  // iree_linalg_ext.scatter (both the array/row-scatter and scalar paths).
  // That dialect is gone now; this needs a stablehlo.scatter-based lowering,
  // landing alongside SegHist's (see LowerSegHist below).
  Values LowerWithAcc(const ExpWithAcc &, Ctx &) { Undefined("with_acc"); }

  Values LowerDotGeneral(mlir::ValueRange args, mlir::TypeRange retTypes,
                         Ctx &ctx) {
    // StableHLO dot_general
    //
    // The following signature is without support for
    // - precision parameters,
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

    auto numNonSizeArgs = 6;
    assert(args.size() >= numNonSizeArgs);
    assert(retTypes.size() == 1);
    auto getArg = [&](int i) {
      // We don't know how many size parameters will be passed,
      // but they are always preprended, so we index from the back.
      return args[args.size() - numNonSizeArgs + i];
    };
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

    auto op = mlir::stablehlo::DotGeneralOp::create(
        builder,
        mlir::RankedTensorType::get(resultShape, lhsType.getElementType()),
        lhs,
        rhs,
        mlir::stablehlo::DotDimensionNumbersAttr::get(&context,
                                                      lhsBatchingDimensions,
                                                      rhsBatchingDimensions,
                                                      lhsContractingDimensions,
                                                      rhsContractingDimensions),
        {},
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

    auto numNonSizeArgs = 9;
    assert(args.size() >= numNonSizeArgs);
    assert(retTypes.size() == 1);
    auto getArg = [&](int i) {
      // We don't know how many size parameters will be passed,
      // but they are always preprended, so we index from the back.
      return args[args.size() - numNonSizeArgs + i];
    };
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

  Values LowerRaggedDotNonUniform(mlir::ValueRange args,
                                  mlir::TypeRange retTypes, Ctx &ctx) {
    // A nonuniform ragged_dot expressed over the flat segmented representation
    // produced by irregular flattening. This blackbox sits on matmul_nonuniform,
    // so it is lowered at its application site where the actual (constant) size
    // arguments are visible: k = ks[0], n = ns[0], b_data static.
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
    //   (r : i64, sizes : [g]i64, flags : [r]i1, offsets : [g]i64, data : [r]f32)
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
    // linalg.broadcast(from_elements(constant)) (see BasicOpReplicate), so peel
    // those to reach the scalar; also accept a folded dense constant directly.
    auto replicatedInt = [&](mlir::Value v) -> int64_t {
      mlir::Value cur = v;
      // Peel any shape-only wrappers (a `:>` size coercion in the source lowers
      // to reshape/cast) to reach the underlying replicate.
      while (auto *op = cur.getDefiningOp()) {
        if (mlir::isa<mlir::tensor::ReshapeOp, mlir::tensor::CastOp,
                      mlir::tensor::ExpandShapeOp, mlir::tensor::CollapseShapeOp>(
                op))
          cur = op->getOperand(0);
        else
          break;
      }
      mlir::DenseIntElementsAttr dense;
      if (mlir::matchPattern(cur, mlir::m_Constant(&dense)))
        return dense.getValues<int64_t>()[0];
      // Futhark's replicate lowers to linalg.broadcast(from_elements(constant)).
      if (auto bcast = cur.getDefiningOp<mlir::linalg::BroadcastOp>())
        cur = bcast.getInput();
      if (auto fe = cur.getDefiningOp<mlir::tensor::FromElementsOp>())
        cur = fe.getOperand(0);
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
    // TEMP: until the flatten threads a_k statically, uncomment to force a static
    // m for the m=128 bench and exercise the static chlo.ragged_dot path.
    // if (mlir::ShapedType::isDynamic(m)) m = 128;

    mlir::Value resultMN; // the [m][n] product
    if (!mlir::ShapedType::isDynamic(m)) {
      // ---- static m: a faithful, minimal chlo.ragged_dot ----
      // rhs: static [g][k][n]; b is not ragged so its flat length is static.
      auto rhsTy = mlir::RankedTensorType::get({g, k, n}, elementTy);
      auto bDataTy = getRankedType(bData.getType());
      mlir::Value rhs =
          bDataTy == rhsTy
              ? bData
              : mlir::tensor::ExpandShapeOp::create(
                    builder, rhsTy, bData,
                    *mlir::getReassociationIndicesForReshape(bDataTy, rhsTy))
                    .getResult();
      // lhs [m][k]: expand_shape needs a statically-sized source, so cast the
      // (possibly dynamic) flat a_data to [m*k] first.
      auto lhsTy = mlir::RankedTensorType::get({m, k}, elementTy);
      auto flatTy = mlir::RankedTensorType::get({m * k}, elementTy);
      mlir::Value flat =
          aDataTy.hasStaticShape()
              ? aData
              : mlir::tensor::CastOp::create(builder, flatTy, aData).getResult();
      mlir::SmallVector<mlir::ReassociationIndices> reassoc{{0, 1}};
      mlir::Value lhs =
          mlir::tensor::ExpandShapeOp::create(builder, lhsTy, flat, reassoc)
              .getResult();
      // Mode 1: ragged dim lhs axis 0; contract lhs axis 1 with rhs axis 1;
      // rhs group dim axis 0; no batching.
      mlir::SmallVector<int64_t> noBatch{};
      resultMN = mlir::chlo::RaggedDotOp::create(
                     builder, mlir::RankedTensorType::get({m, n}, elementTy),
                     lhs, rhs, ms,
                     mlir::chlo::RaggedDotDimensionNumbersAttr::get(
                         &context, noBatch, noBatch,
                         mlir::SmallVector<int64_t>{1},
                         mlir::SmallVector<int64_t>{1},
                         mlir::SmallVector<int64_t>{0},
                         mlir::SmallVector<int64_t>{0}),
                     {})
                     ->getResult(0);
    } else {
      // ---- dynamic m: the dense masked dot_general strategy (gg/ragged_matmul).
      // IREE cannot legalize chlo.ragged_dot's internal dynamic_broadcast for a
      // runtime m, but stablehlo.dot_general handles dynamic dims, and the mask
      // is a single affine linalg.generic (no dynamic broadcast).
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

      // masked [m][g][k]: masked[i,e,j] = lhs2d[i,j] if row i in group e else 0,
      // where group e spans rows [a_offsets[e]/k, a_offsets[e]/k + ms[e]).
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
      auto kI64 =
          mlir::arith::ConstantOp::create(builder, builder.getI64IntegerAttr(k));
      auto zeroElem = mlir::arith::ConstantOp::create(
          builder, builder.getZeroAttr(elementTy));
      mlir::Value masked =
          mlir::linalg::GenericOp::create(
              builder, mlir::TypeRange{maskedTy},
              mlir::ValueRange{lhs2d, aOffsets, ms},
              mlir::ValueRange{maskedInit}, maps, iters,
              [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange a) {
                mlir::OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPointToEnd(b.getInsertionBlock());
                auto iIdx = mlir::linalg::IndexOp::create(builder, loc, 0);
                auto iI64 = mlir::arith::IndexCastOp::create(
                    builder, loc, builder.getI64Type(), iIdx);
                auto start =
                    mlir::arith::DivUIOp::create(builder, loc, a[1], kI64);
                auto end = mlir::arith::AddIOp::create(builder, loc, start, a[2]);
                auto ge = mlir::arith::CmpIOp::create(
                    builder, loc, mlir::arith::CmpIPredicate::sge, iI64, start);
                auto lt = mlir::arith::CmpIOp::create(
                    builder, loc, mlir::arith::CmpIPredicate::slt, iI64, end);
                auto inR = mlir::arith::AndIOp::create(builder, loc, ge, lt);
                auto val = mlir::arith::SelectOp::create(builder, loc, inR, a[0],
                                                         zeroElem);
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
              builder, rhsFusedTy, bData,
              *mlir::getReassociationIndicesForReshape(bDataTy, rhsFusedTy))
              .getResult();
      resultMN = mlir::stablehlo::DotGeneralOp::create(
                     builder, mlir::RankedTensorType::get({dyn, n}, elementTy),
                     masked2d, rhsFused,
                     mlir::stablehlo::DotDimensionNumbersAttr::get(
                         &context, /*lhsBatch=*/{}, /*rhsBatch=*/{},
                         /*lhsContract=*/{1}, /*rhsContract=*/{0}),
                     {}, {})
                     .getResult();
    }

    // data: flatten [m][n] -> [m*n].
    auto dataTy = mlir::RankedTensorType::get(
        {mlir::ShapedType::isDynamic(m) ? dyn : m * n}, elementTy);
    mlir::SmallVector<mlir::ReassociationIndices> dataReassoc{{0, 1}};
    mlir::Value data = mlir::tensor::CollapseShapeOp::create(
                           builder, dataTy, resultMN, dataReassoc)
                           .getResult();

    // r = length of data (= m*n), as i64.
    auto rIdx = mlir::tensor::DimOp::create(builder, data, 0).getResult();
    mlir::Value r = mlir::arith::IndexCastOp::create(
                        builder, builder.getI64Type(), rIdx)
                        .getResult();

    // Dummy auxiliary segment structures.
    auto i64Ty = builder.getI64Type();
    auto gI64Ty = mlir::RankedTensorType::get({g}, i64Ty);
    auto zeros = mlir::DenseElementsAttr::get(gI64Ty, static_cast<int64_t>(0));
    mlir::Value sizes = mlir::arith::ConstantOp::create(builder, zeros);
    mlir::Value offsets = mlir::arith::ConstantOp::create(builder, zeros);
    auto flagsTy = mlir::RankedTensorType::get({dyn}, builder.getI1Type());
    auto flagsEmpty = mlir::tensor::EmptyOp::create(builder, flagsTy,
                                                    mlir::ValueRange{rIdx});
    auto trueC =
        mlir::arith::ConstantOp::create(builder, builder.getBoolAttr(true));
    mlir::Value flags = mlir::linalg::FillOp::create(
                            builder, mlir::ValueRange{trueC},
                            mlir::ValueRange{flagsEmpty})
                            .result();

    return {r, sizes, flags, offsets, data};
  }

  Values LowerArgSort(mlir::ValueRange args, mlir::TypeRange retTypes,
                      Ctx &ctx) {
    // A basic ascending sort.
    // Doesn't implement StableHLO sort faithfully.
    auto numNonSizeArgs = 2;
    assert(args.size() >= numNonSizeArgs);
    assert(retTypes.size() == 1);
    auto getArg = [&](int i) {
      // We don't know how many size parameters will be passed,
      // but they are always preprended, so we index from the back.
      return args[args.size() - numNonSizeArgs + i];
    };
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
        mlir::stablehlo::ConstantOp::create(
            builder, mlir::DenseElementsAttr::get(
                         i64Scalar, builder.getI64IntegerAttr(0))),
        mlir::stablehlo::ConstantOp::create(
            builder, mlir::DenseElementsAttr::get(
                         i64Scalar, builder.getI64IntegerAttr(1))),
        builder.getI64Type(),
        inputsTy.getShape()[0]);

    // Sort the values and the indices together, keyed on the values; the
    // permuted indices are the argsort result.
    auto op = mlir::stablehlo::SortOp::create(
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
      auto lt = mlir::stablehlo::CompareOp::create(
          builder,
          block->getArgument(0),
          block->getArgument(1),
          mlir::stablehlo::ComparisonDirection::LT,
          mlir::stablehlo::ComparisonType::SIGNED);
      mlir::stablehlo::ReturnOp::create(builder, mlir::ValueRange{lt});
    }

    return {op.getResult(1)};
  }

  mlir::RankedTensorType getRankedType(const mlir::Type &ty) {
    if (auto t = mlir::dyn_cast<mlir::RankedTensorType>(ty))
      return t;
    Undefined();
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
    mlir::Value base = mlir::stablehlo::IotaOp::create(
        builder, returnTy, builder.getI64IntegerAttr(0));
    auto bx = mlir::stablehlo::BroadcastInDimOp::create(
        builder, returnTy, x, mlir::SmallVector<int64_t>{});
    auto bs = mlir::stablehlo::BroadcastInDimOp::create(
        builder, returnTy, s, mlir::SmallVector<int64_t>{});
    auto scaled = mlir::stablehlo::MulOp::create(builder, base, bs);
    return mlir::stablehlo::AddOp::create(builder, scaled, bx);
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
  mlir::Value LowerIndex(const BasicOpIndex &index, Ctx &ctx) {
    auto tensor = LowerSubExp(index.vName.name, ctx);

    auto isSlice = [](const DimIndex<SubExp> &d) {
      return std::holds_alternative<DimSlice<SubExp>>(d.v);
    };
    if (!llvm::any_of(index.slice.dims, isSlice)) {
      // stablehlo.dynamic_slice handles constant and dynamic indices
      // uniformly (its start_indices are themselves 0-d tensors already, no
      // unwrapping needed), then stablehlo.reshape drops the resulting
      // all-ones shape down to a 0-d scalar.
      Values startIndices;
      for (const auto &dim : index.slice.dims)
        startIndices.push_back(
            LowerSubExp(std::get<DimFix<SubExp>>(dim.v).i, ctx));
      auto tensorTy = getRankedType(tensor.getType());
      mlir::SmallVector<int64_t> sliceSizes(tensorTy.getRank(), 1);
      auto sliceTy =
          mlir::RankedTensorType::get(sliceSizes, tensorTy.getElementType());
      auto slice = mlir::stablehlo::DynamicSliceOp::create(
          builder, sliceTy, tensor, startIndices, sliceSizes);
      return mlir::stablehlo::ReshapeOp::create(
          builder, mlir::RankedTensorType::get({}, tensorTy.getElementType()),
          slice);
    }

    // Constants stay attributes so that they remain visible in the shape of
    // the extracted slice.
    auto toIndex = [&](const SubExp &e) -> mlir::OpFoldResult {
      if (auto *c = std::get_if<ConstantSubExp>(&e.v))
        return builder.getIndexAttr(c->GetIntValue());
      auto bare = mlir::tensor::ExtractOp::create(builder, LowerSubExp(e, ctx),
                                                   mlir::ValueRange{});
      return mlir::OpFoldResult(
          mlir::arith::IndexCastOp::create(builder, builder.getIndexType(),
                                           bare));
    };

    mlir::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;
    std::vector<int64_t> resultShape;
    for (const auto &dim : index.slice.dims) {
      match(
          dim,
          [&](const DimFix<SubExp> &fix) {
            offsets.push_back(toIndex(fix.i));
            sizes.push_back(builder.getIndexAttr(1));
            strides.push_back(builder.getIndexAttr(1));
          },
          [&](const DimSlice<SubExp> &s) {
            offsets.push_back(toIndex(s.start));
            sizes.push_back(toIndex(s.length));
            // tensor.extract_slice cannot step backwards, so a stride we
            // cannot see to be positive -- a reversal, say -- is out of reach.
            auto *stride = std::get_if<ConstantSubExp>(&s.stride.v);
            if (!stride || stride->GetIntValue() < 1)
              Undefined();
            strides.push_back(builder.getIndexAttr(stride->GetIntValue()));
            resultShape.push_back(toShapeType(s.length));
          });
    }

    auto resultTy = mlir::RankedTensorType::get(
        resultShape, getRankedType(tensor.getType()).getElementType());
    return mlir::tensor::ExtractSliceOp::create(
        builder, resultTy, tensor, offsets, sizes, strides);
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
          auto concat =
              mlir::tensor::ConcatOp::create(builder, val.dim, operands);
          return concat->getResult(0);
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
                return mlir::stablehlo::ConvertOp::create(
                    builder,
                    LowerTy(Type::CreatePrim(PrimType::Int(sext.to))),
                    op0);
              },
              [&](const ConvOpSIToFP &sitofp) -> mlir::Value {
                auto ty = mlir::RankedTensorType::get(
                    {}, GetFloatType(builder, GetWidth(sitofp.to)));
                return mlir::stablehlo::ConvertOp::create(builder, ty, op0);
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
          return mlir::stablehlo::ReshapeOp::create(builder, resultTy, op0);
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
          return mlir::stablehlo::TransposeOp::create(builder, transposedTy,
                                                       op, perm);
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
          if (llvm::any_of(before, mlir::ShapedType::isDynamic))
            Undefined("replicate: dynamic shape");
          auto original = tensorTy.getShape();
          before.insert(before.end(), original.begin(), original.end());
          auto resultTy =
              mlir::RankedTensorType::get(before, tensorTy.getElementType());
          mlir::SmallVector<int64_t> broadcastDims;
          int64_t rankDiff = (int64_t)before.size() - (int64_t)original.size();
          for (int64_t i = 0; i < (int64_t)original.size(); ++i)
            broadcastDims.push_back(rankDiff + i);
          return mlir::stablehlo::BroadcastInDimOp::create(
              builder, resultTy, op, broadcastDims);
        },
        [&](const BasicOpScratch &val) -> mlir::Value {
          auto baseTy = LowerPrimType(val.type);
          auto shapeTy = toShapeType(val.dims);
          Values dynamicSizes;
          for (auto [d, t] : llvm::zip_equal(val.dims, shapeTy)) {
            if (mlir::ShapedType::isDynamic(t)) {
              auto v = LowerSubExp(d, ctx);
              dynamicSizes.push_back(mlir::arith::IndexCastOp::create(
                  builder, builder.getIndexType(), v));
            }
          }
          return mlir::tensor::EmptyOp::create(
              builder,
              mlir::RankedTensorType::get(shapeTy, baseTy),
              dynamicSizes);
        },
        [&](const BasicOpCmpOp &val) -> mlir::Value {
          auto lhs = LowerSubExp(val.op0, ctx);
          auto rhs = LowerSubExp(val.op1, ctx);
          using CD = mlir::stablehlo::ComparisonDirection;
          using CT = mlir::stablehlo::ComparisonType;
          auto cmp = [&](CD dir, CT ty) -> mlir::Value {
            return mlir::stablehlo::CompareOp::create(builder, lhs, rhs, dir,
                                                      ty);
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
                  return mlir::stablehlo::NotOp::create(builder, v);
                return mlir::stablehlo::NegOp::create(builder, v);
              },
              [&](const UnOpComplement &) -> mlir::Value {
                return mlir::stablehlo::NotOp::create(builder, v);
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
            // result into a size-1 array.
            auto lifted = mlir::RankedTensorType::get({1}, LowerTy(ty));
            results.push_back(mlir::tensor::FromElementsOp::create(
                                  builder, lifted, mlir::ValueRange{v})
                                  .getResult());
          }
          return results;
        });
  }

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
  Values LowerSegMap(const SegMap &pSegMap, Ctx &ctx) {
    return LowerSegMap(
        pSegMap.lvl, pSegMap.space, pSegMap.body, pSegMap.ret, ctx);
  }

  // Lowers a SegMap to a stablehlo.map. Operands are the kernel's affine
  // reads plus one full-shape stablehlo.iota per iteration-space dimension;
  // the region is then made isolated-from-above (stablehlo.map's region
  // can't capture outer SSA values, per the StableHLO/XLA target model) by
  // discovering every other outer value it ends up using, broadcasting it to
  // the full shape, and appending it as an extra operand -- rather than
  // hand-rolling a Futhark-AST free-variable walker, this reuses MLIR's own
  // mlir::getUsedValuesDefinedAbove/replaceAllUsesInRegionWith on the
  // already-lowered region.
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
        arr = mlir::stablehlo::BroadcastInDimOp::create(builder, broadcastTy,
                                                        arr, broadcastDims);
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
      operands.push_back(mlir::stablehlo::IotaOp::create(
          builder, iotaTy, builder.getI64IntegerAttr(d.index)));
    }

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
      auto op = mlir::stablehlo::MapOp::create(
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
                 iterSpace, blockArgs.drop_front(affine_reads.size())))
          local.subexps.insert(d.id, arg);

        std::unordered_set<std::string> skipLowering;
        for (auto &read : affine_reads)
          skipLowering.insert(read.result.name);
        for (const Stm &stm : body.stms) {
          assert(stm.pat.elems.size() == 1);
          if (skipLowering.contains(stm.pat.elems[0].name.name))
            continue;
          LowerStm(stm, local);
        }
        mlir::Value result = LowerSubExp(body.result[j].result, local);
        mlir::stablehlo::ReturnOp::create(builder, mlir::ValueRange{result});
      }

      // mlir::makeRegionIsolatedFromAbove needs an mlir::IRRewriter, whose
      // base class mlir::RewriterBase has no typeinfo in the prebuilt MLIR
      // libs (LLVM/MLIR defaults to -fno-rtti; this backend builds with
      // -frtti), which fails at link time. Doing the same thing with the
      // two plain free functions it's built from avoids needing a
      // RewriterBase instance.
      mlir::SetVector<mlir::Value> freeVars;
      mlir::getUsedValuesDefinedAbove(op.getRegion(), freeVars);

      if (!freeVars.empty()) {
        mlir::Block &entry = op.getRegion().front();
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(op);
        Values broadcasts;
        for (mlir::Value fv : freeVars) {
          auto fvTy = mlir::dyn_cast<mlir::RankedTensorType>(fv.getType());
          if (!fvTy || fvTy.getRank() != 0)
            Undefined("SegMap: non-scalar free variable");
          auto broadcastTy =
              mlir::RankedTensorType::get(shapeTy, fvTy.getElementType());
          broadcasts.push_back(mlir::stablehlo::BroadcastInDimOp::create(
              builder, broadcastTy, fv, mlir::SmallVector<int64_t>{}));
          mlir::BlockArgument newArg =
              entry.addArgument(fv.getType(), builder.getLoc());
          mlir::replaceAllUsesInRegionWith(fv, newArg, op.getRegion());
        }
        op->insertOperands(op->getNumOperands(), broadcasts);
      }

      finalResults.push_back(op->getResult(0));
    }

    return finalResults;
  }

  // TODO Same limitations as SegMap, probably.
  // segred = segmap (the fused per-element part, reused as-is) + a
  // stablehlo.reduce over the innermost (segment) dimension. Tuple
  // operators fall out for free -- stablehlo.reduce is natively variadic
  // (multiple inputs/init_values/region-results), no special-casing needed
  // beyond what's already generic below.
  Values LowerSegRed(const SegRed &pSegRed, Ctx &ctx) {
    RequireThreadLevel(pSegRed.lvl);

    if (pSegRed.ops.size() != 1)
      Undefined("segred: multiple operators");
    auto pSegBinOp = pSegRed.ops[0];
    if (pSegBinOp.shape.dims.size() != 0)
      Undefined("segred: vector-valued operator");
    // The StableHLO spec says reduce preserves element order, but XLA's GPU
    // reduce may assume commutativity -- lower priority to actually solve
    // (see Phase 8), so flagged rather than risking a silently wrong result.
    if (pSegBinOp.comm == Commutativity::Noncommutative)
      Undefined("segred: non-commutative");

    IterationSpace iterSpace = LowerSegSpace(pSegRed.space, ctx);
    int64_t rank = std::ssize(iterSpace);
    if (rank == 0)
      Undefined("segred: rank 0");

    Values mapped = LowerSegMap(pSegRed.lvl, pSegRed.space, pSegRed.body,
                                pSegRed.ret, ctx);

    Values neutral;
    for (auto &e : pSegBinOp.neutral)
      neutral.push_back(LowerSubExp(e, ctx));
    int64_t valueCount = (int64_t)neutral.size();

    mlir::SmallVector<int64_t> reduceDims{rank - 1};
    mlir::SmallVector<mlir::Type> resultTypes;
    for (mlir::Value v : mapped) {
      auto ty = getRankedType(v.getType());
      resultTypes.push_back(
          mlir::RankedTensorType::get(ty.getShape().drop_back(),
                                      ty.getElementType()));
    }

    auto op = mlir::stablehlo::ReduceOp::create(builder, resultTypes, mapped,
                                                neutral, reduceDims);

    // Per the StableHLO spec, the body's block args are (one value per input
    // slice, then one accumulator per init_value) -- the *opposite* order
    // from Futhark's own SegBinOp.lambda.params convention (accumulators
    // first, then new values).
    mlir::SmallVector<mlir::Type> argTys;
    for (mlir::Value v : mapped)
      argTys.push_back(mlir::RankedTensorType::get(
          {}, getRankedType(v.getType()).getElementType()));
    for (mlir::Value v : neutral)
      argTys.push_back(v.getType());
    mlir::SmallVector<mlir::Location> locs(argTys.size(), builder.getLoc());

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      mlir::Block *blk =
          builder.createBlock(&op.getBody(), {}, argTys, locs);
      builder.setInsertionPointToStart(blk);

      Ctx local = ctx;
      mlir::ValueRange blockArgs = blk->getArguments();
      mlir::ValueRange newValueArgs = blockArgs.take_front(valueCount);
      mlir::ValueRange accArgs = blockArgs.drop_front(valueCount);
      for (auto [param, acc] : llvm::zip_equal(
               pSegBinOp.lambda.params | std::views::take(valueCount),
               accArgs))
        local.subexps.insert(param.name, acc);
      for (auto [param, nv] : llvm::zip_equal(
               pSegBinOp.lambda.params | std::views::drop(valueCount),
               newValueArgs))
        local.subexps.insert(param.name, nv);

      Values results = LowerBody(pSegBinOp.lambda.body, local);
      mlir::stablehlo::ReturnOp::create(builder, results);
    }

    return op->getResults();
  }

  // TODO(StableHLO rewrite, Phase 7): was IREE's iree_linalg_ext.scatter.
  // Needs a stablehlo.scatter-based lowering (see prompt.md's SegHist
  // mapping): operand = dest, scatter_indices = indices, updates = values,
  // update_computation = the HistOp's combining lambda.
  Values LowerSegHist(const SegHist &, Ctx &) { Undefined("SegHist"); }

  // TODO(StableHLO rewrite, Phase 5/6): was IREE's iree_linalg_ext.scan (with
  // a vector<Nxi64>-lane packing trick for tuple carries, worked around an
  // IREE-specific bug -- not relevant to the new backend). Needs a
  // chlo.scan-based lowering first (scalar, associative), then a
  // slice/pad/concatenate parallel-prefix fallback for tuple/non-scalar
  // operators, per prompt.md's segscan mapping.
  Values LowerSegScan(const SegScan &, Ctx &) { Undefined("SegScan"); }

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
    return mlir::stablehlo::ConstantOp::create(
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
    if (std::get_if<BinOpSDivUp>(&binOp.v) || std::get_if<BinOpSDiv>(&binOp.v) ||
        std::get_if<BinOpUDiv>(&binOp.v) || std::get_if<BinOpUDivUp>(&binOp.v))
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
      return mlir::stablehlo::ConstantOp::create(
          builder,
          mlir::DenseElementsAttr::get(getRankedType(tensorTy), attrs));
    } else {
      // TODO(StableHLO rewrite): tensor.from_elements is banned by XLA; a
      // non-constant array literal needs concatenate of reshaped (0-d -> 1-d)
      // elements instead. Not exercised by any currently-passing test.
      auto values = mlir::SmallVector<mlir::Value>();
      for (auto subExp : arrayLit.values) {
        values.push_back(LowerSubExp(subExp, ctx));
      }
      return mlir::tensor::FromElementsOp::create(builder, tensorTy, values);
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
      return {mlir::stablehlo::SelectOp::create(builder, args[0], args[1],
                                                args[2])
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

  std::vector<int64_t> LowerShape(const Shape &shape) {
    return toShapeType(shape.dims);
  }

  mlir::Type LowerTy(Type t) {
    auto baseTy = &t.t.v;
    if (auto *val = std::get_if<TypePrim<Shape, NoUniqueness>>(baseTy)) {
      // Scalars are 0-d tensors (a hard XLA requirement), unlike an array's
      // own element type (see LowerTypeArray, which calls LowerPrimType
      // directly to get the bare element type instead).
      return mlir::RankedTensorType::get({}, LowerPrimType(val->t));
    }

    if (auto *val = std::get_if<TypeArray<Shape, NoUniqueness>>(baseTy)) {
      return LowerTypeArray(*val);
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

  mlir::Type LowerTypeArray(TypeArray<Shape, NoUniqueness> typeArray) {
    return mlir::RankedTensorType::get(LowerShape(typeArray.shape),
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
