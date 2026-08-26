#pragma once
// Represents the state of a Futhark function
#include "blackbox.hpp"
#include "debug.hpp"
#include "ir/core.hpp"
#include "ir/segop.hpp"
#include "ir/syntax.hpp"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
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
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
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
    context.getOrLoadDialect<mlir::iree_compiler::IREE::Flow::FlowDialect>();
    context.getOrLoadDialect<
        mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect>();
    context.getOrLoadDialect<mlir::tensor::TensorDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::index::IndexDialect>();
    context.getOrLoadDialect<mlir::BuiltinDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
    context.getOrLoadDialect<mlir::chlo::ChloDialect>();
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

    auto fType = builder.getFunctionType(inputTypes, retTypes);
    auto func =
        mlir::func::FuncOp::create(builder.getUnknownLoc(), fun.name, fType);
    builder.insert(func);
    functions[fun] = func;

    if (!fun.entry) {
      func.setPrivate();
    }

    // Annotate function arguments with size type information
    // by mapping tensor dimensions to function argument positions.
    //
    // For example, our test runner must infer the values of size parameters:
    //   -- ==
    //   -- input { [...] }
    //   -- output { ... }
    //   entry f [n] [m] (x: [n][10][m]f32)
    // has GPU IR with additional arguments n and m
    //   entry ("f", {n : i64, m : i64, x : [n][10][m]i64}, ...)
    // so we annotate x with "futhark.size_args = [0, -1, 1]" in MLIR
    // where -1 denotes a static dimension.
    if (fun.entry) {
      std::unordered_map<std::string, int> argPosition;
      for (auto [i, param] : llvm::enumerate(runtimeParams)) {
        argPosition[param.name] = i;

        if (auto *arr =
                std::get_if<TypeArray<Shape, NoUniqueness>>(&param.dec.v.t.v)) {
          mlir::SmallVector<int64_t> pos;
          for (const auto &d : arr->shape.dims)
            pos.push_back(match(
                d,
                [&](const VarSubExp &) {
                  return argPosition.at(d.GetVName().name);
                },
                [&](const ConstantSubExp &) { return -1; }));

          func.setArgAttr(i, "futhark.size_args", builder.getI64ArrayAttr(pos));
        }
      }
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
                case BlackBox::ArgSort:
                  return LowerArgSort(args, retTypes, ctx);
                case BlackBox::Scatter:
                  return LowerScatter(args, retTypes, ctx);
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
          return mlir::scf::IfOp::create(
                     builder,
                     LowerSubExp(e.cond, ctx),
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

  // A Futhark loop carries a set of parameters from one iteration to
  // the next, which become the iteration arguments of the scf.for.
  Values LowerLoop(const ExpLoop &loop, Ctx &ctx) {
    // TODO while loops, which need an scf.while with the continuation
    // condition -- itself a loop parameter -- read in the "before" region.
    auto *forLoop = std::get_if<ForLoop>(&loop.form.v);
    if (!forLoop)
      Undefined();

    mlir::SmallVector<mlir::Type> mergeTypes;
    Values inits;
    for (const auto &[param, init] : loop.merge) {
      auto ty = LowerTy(param.dec.v);
      mergeTypes.push_back(ty);
      inits.push_back(castToType(LowerSubExp(init, ctx), ty));
    }

    auto lb = mlir::arith::ConstantIndexOp::create(builder, 0);
    auto ub = mlir::arith::IndexCastOp::create(
        builder, builder.getIndexType(), LowerSubExp(forLoop->bound, ctx));
    auto step = mlir::arith::ConstantIndexOp::create(builder, 1);
    auto counterTy = builder.getIntegerType(GetWidth(forLoop->t));

    auto forOp = mlir::scf::ForOp::create(
        builder,
        builder.getLoc(),
        lb,
        ub,
        step,
        inits,
        [&](mlir::OpBuilder &b,
            mlir::Location bodyLoc,
            mlir::Value iv,
            mlir::ValueRange iterArgs) {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(b.getInsertionBlock(),
                                    b.getInsertionPoint());

          Ctx local = ctx;
          // scf.for counts in `index`, but the body sees the Futhark loop
          // counter at its own integer type.
          local.subexps.insert(forLoop->i.name,
                               mlir::arith::IndexCastOp::create(
                                   builder, bodyLoc, counterTy, iv));
          for (auto [param, arg] : llvm::zip_equal(loop.merge, iterArgs))
            local.subexps.insert(param.first.name, arg);

          // The body's results feed back into the merge parameters, so they
          // must have exactly the parameters' types.
          Values results;
          for (auto [v, ty] :
               llvm::zip_equal(LowerBody(*loop.body, local), mergeTypes))
            results.push_back(castToType(v, ty));
          mlir::scf::YieldOp::create(builder, bodyLoc, results);
        });

    return forOp.getResults();
  }

  // Casts the type of `value` to `type` if they are compatible
  // (e.g., a tensor's static dimension is cast to a dynamic one
  // or a 0-D tensor is cast to a scalar).
  mlir::Value castToType(mlir::Value value, mlir::Type type) {
    if (value.getType() == type) {
      return value;
    }
    if (mlir::tensor::CastOp::areCastCompatible(value.getType(), type)) {
      return mlir::tensor::CastOp::create(builder, type, value);
    }
    if (mlir::isa<mlir::RankedTensorType>(value.getType()) &&
        getRankedType(value.getType()).getRank() == 0 && type.isIntOrFloat()) {
      return mlir::tensor::ExtractOp::create(builder, value, {});
    }
    llvm::errs() << "error: tried to cast\n\n"
                 << value << "\n\nto type\n"
                 << type << "\n";
    Undefined();
  }

  Values LowerMatMul(mlir::ValueRange args, mlir::TypeRange retTypes,
                     Ctx &ctx) {
    // Signature is (d0, d1, d3, A : [d4][d5]t, B : [d5][d6]t) -> [d4][d6]t
    // Where d0 d1 d3 are equal to d4 d5 d6 somehow, depending on the order
    // specified by the user in the Futhark source/IR. So we disregard the size
    // parameters entirely and grab them from the matrix arguments A and B.
    assert(args.size() == 5);
    assert(retTypes.size() == 1);
    auto x = args[3];
    auto y = args[4];
    auto m = mlir::tensor::DimOp::create(builder, x, 0).getResult();
    auto n = mlir::tensor::DimOp::create(builder, y, 1).getResult();

    // Create the output matrix.
    auto z_type = getRankedType(retTypes[0]);
    Values dynamicSizes;
    for (auto [d, t] :
         llvm::zip_equal(std::vector<mlir::Value>{m, n}, z_type.getShape())) {
      if (mlir::ShapedType::isDynamic(t))
        dynamicSizes.push_back(d);
    }
    auto zero = mlir::arith::ConstantOp::create(
        builder, builder.getZeroAttr(z_type.getElementType()));
    auto z = mlir::linalg::FillOp::create(
                 builder,
                 mlir::ValueRange{zero},
                 mlir::ValueRange{mlir::tensor::EmptyOp::create(
                     builder, z_type, dynamicSizes)})
                 .result();

    return mlir::linalg::MatmulOp::create(builder,
                                          builder.getLoc(),
                                          mlir::ValueRange{x, y},
                                          mlir::ValueRange{z})
        .getResults();
  }

  Values LowerUpdateAcc(const BasicOpUpdateAcc &update, Ctx &ctx) {
    assert(update.indices.size() == 1);
    assert(update.values.size() == 1);
    mlir::Value index = LowerSubExp(update.indices[0], ctx);
    mlir::Value value = LowerSubExp(update.values[0], ctx);
    return {value, index};
  }

  mlir::Value EmitScatter(mlir::Value update, mlir::Value index,
                          mlir::Value original) {
    namespace LinalgExt = mlir::iree_compiler::IREE::LinalgExt;

    auto originalType = mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
    if (!originalType)
      Undefined();

    llvm::SmallVector<int64_t> dimensionMap{0};
    auto scatter = LinalgExt::ScatterOp::create(
        builder,
        mlir::TypeRange{originalType},
        update,
        index,
        /*mask=*/mlir::Value(),
        original,
        dimensionMap,
        /*unique_indices=*/false);

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      mlir::Type elementType = originalType.getElementType();
      mlir::SmallVector<mlir::Type> argumentTypes{elementType, elementType};
      mlir::SmallVector<mlir::Location> argumentLocations(2, builder.getLoc());
      mlir::Block *body = builder.createBlock(&scatter.getRegion(), {}, argumentTypes, argumentLocations);
      builder.setInsertionPointToStart(body);
      LinalgExt::YieldOp::create(builder, body->getArgument(0));
    }

    return scatter.getResult(0);
  }

  Values LowerWithAcc(const ExpWithAcc &withAcc, Ctx &ctx) {
    assert(withAcc.inputs.size() == 1);
    const WithAccInput &input = withAcc.inputs[0];
    assert(input.arrays.size() == 1);
    assert(!input.op.has_value());

    const Lambda &lambda = *withAcc.lambda;
    Ctx local = ctx;
    for (const Stm &stm : lambda.body.stms)
      LowerStm(stm, local);

    assert(lambda.body.result.size() >= withAcc.inputs.size());
    VName returnedAcc = lambda.body.result[0].GetVName();
    AccValue acc = local.accs.lookup(returnedAcc.name);
    mlir::Value original = ctx.subexps.lookup(input.arrays[0].name);

    Values results{EmitScatter(acc.update, acc.index, original)};
    for (size_t i = withAcc.inputs.size(); i < lambda.body.result.size(); ++i)
      results.push_back(LowerSubExp(lambda.body.result[i], local));
    return results;
  }

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

    mlir::Value indices = createIota(
        mlir::tensor::DimOp::create(builder, inputs, 0).getResult(),
        mlir::arith::ConstantOp::create(builder, builder.getI64IntegerAttr(0)),
        mlir::arith::ConstantOp::create(builder, builder.getI64IntegerAttr(1)),
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

  Values LowerScatter(mlir::ValueRange args, mlir::TypeRange retTypes,
                      Ctx &ctx) {
    // TODO Remove black box scatter once we handle WithAcc.
    auto numNonSizeArgs = 3;
    assert(args.size() >= numNonSizeArgs);
    assert(retTypes.size() == 1);
    auto getArg = [&](int i) {
      // We don't know how many size parameters will be passed,
      // but they are always preprended, so we index from the back.
      return args[args.size() - numNonSizeArgs + i];
    };
    auto dest = getArg(0);
    auto indices = getArg(1);
    auto values = getArg(2);

    auto elementTy = getRankedType(values.getType()).getElementType();

    auto op = mlir::iree_compiler::IREE::LinalgExt::ScatterOp::create(
        builder, dest.getType(), values, indices, {}, dest, {0}, false);

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto *blk = builder.createBlock(&op.getRegion(),
                                      {},
                                      {elementTy, elementTy},
                                      {builder.getLoc(), builder.getLoc()});
      mlir::iree_compiler::IREE::LinalgExt::YieldOp::create(
          builder, blk->getArgument(0));
    }

    return {op.getResult(0)};
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
    Undefined();
  }

  // Builds a 1-D iota tensor where result[i] = x + i*s.
  mlir::Value createIota(mlir::Value n, mlir::Value x, mlir::Value s,
                         mlir::Type elementTy, int64_t dimTy) {
    auto returnTy = mlir::RankedTensorType::get({dimTy}, elementTy);
    auto output = mlir::tensor::EmptyOp::create(
        builder,
        returnTy,
        mlir::ShapedType::isDynamic(dimTy) ? mlir::ValueRange{n}
                                           : mlir::ValueRange{});
    auto op = mlir::linalg::GenericOp::create(
        builder,
        {returnTy},
        {},
        {output},
        {mlir::AffineMap::getMultiDimIdentityMap(1, &context)},
        {mlir::utils::IteratorType::parallel},
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToEnd(b.getInsertionBlock());

          auto i = mlir::arith::IndexCastOp::create(
              builder,
              loc,
              elementTy,
              mlir::linalg::IndexOp::create(builder, loc, 0));

          auto offset = mlir::arith::MulIOp::create(builder, loc, i, s);
          auto result = mlir::arith::AddIOp::create(builder, loc, x, offset);

          mlir::linalg::YieldOp::create(builder, loc, {result});
        });

    return op.getResult(0);
  }

  // `arr[i, j]` reads a single element, but a slice among the indices makes the
  // result an array: `arr[i, a :+ n * s]` becomes a tensor.extract_slice whose
  // fixed dimensions are rank-reduced away.
  mlir::Value LowerIndex(const BasicOpIndex &index, Ctx &ctx) {
    auto tensor = LowerSubExp(index.vName.name, ctx);

    auto isSlice = [](const DimIndex<SubExp> &d) {
      return std::holds_alternative<DimSlice<SubExp>>(d.v);
    };
    if (!llvm::any_of(index.slice.dims, isSlice)) {
      std::vector<mlir::Value> indices;
      for (const auto &dim : index.slice.dims) {
        auto i = LowerSubExp(std::get<DimFix<SubExp>>(dim.v).i, ctx);
        indices.push_back(mlir::arith::IndexCastOp::create(
            builder, builder.getIndexType(), i));
      }
      return mlir::tensor::ExtractOp::create(builder, tensor, indices)
          .getResult();
    }

    // Constants stay attributes so that they remain visible in the shape of
    // the extracted slice.
    auto toIndex = [&](const SubExp &e) -> mlir::OpFoldResult {
      if (auto *c = std::get_if<ConstantSubExp>(&e.v))
        return builder.getIndexAttr(c->GetIntValue());
      return mlir::OpFoldResult(mlir::arith::IndexCastOp::create(
          builder, builder.getIndexType(), LowerSubExp(e, ctx)));
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
          mlir::Value n = mlir::arith::IndexCastOp::create(
              builder, builder.getIndexType(), LowerSubExp(val.n, ctx));
          mlir::Value x = LowerSubExp(val.x, ctx);
          mlir::Value s = LowerSubExp(val.s, ctx);
          auto elementTy = LowerTy(Type::CreatePrim(PrimType::Int(val.t)));
          return createIota(n, x, s, elementTy, toShapeType(val.n));
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
              [&](const ConvOpZExt &zext) -> mlir::Value {
                return mlir::arith::ExtUIOp::create(
                    builder,
                    LowerTy(Type::CreatePrim(PrimType::Int(zext.to))),
                    op0);
              },

              [&](const ConvOpSExt &sext) -> mlir::Value {
                return mlir::arith::ExtSIOp::create(
                    builder,
                    LowerTy(Type::CreatePrim(PrimType::Int(sext.to))),
                    op0);
              },
              [&](const ConvOpSIToFP &sitofp) -> mlir::Value {
                return mlir::arith::SIToFPOp::create(
                    builder, GetFloatType(builder, GetWidth(sitofp.to)), op0);
              },
              [](const auto &) -> mlir::Value { Undefined(); });
        },
        [&](const BasicOpReshape &val) -> mlir::Value {
          auto op0 = LowerSubExp(val.op0, ctx);
          auto sourceTy = getRankedType(op0.getType());

          // `remainder` is the whole shape the reshape produces; the splice
          // fields only say which dimensions were rewritten to get there. A
          // splice may either merge dimensions or split them, so the direction
          // comes from comparing the two ranks.
          auto resultTy = mlir::RankedTensorType::get(
              toShapeType(val.remainder.dims), sourceTy.getElementType());
          if (sourceTy == resultTy)
            return op0;

          auto reassociation =
              mlir::getReassociationIndicesForReshape(sourceTy, resultTy);
          if (!reassociation)
            Undefined();

          if (resultTy.getRank() < sourceTy.getRank())
            return mlir::tensor::CollapseShapeOp::create(
                builder, resultTy, op0, *reassociation);
          return mlir::tensor::ExpandShapeOp::create(
              builder, resultTy, op0, *reassociation);
        },
        [&](const BasicOpRearrange &val) -> mlir::Value {
          auto op = LowerSubExp(val.arr.name, ctx);
          auto tensorTy = llvm::dyn_cast<mlir::RankedTensorType>(op.getType());
          if (!tensorTy)
            Undefined();
          if (val.perm.size() != tensorTy.getShape().size())
            Undefined();

          // linalg.transpose puts input dimension perm[i] in result
          // dimension i, which is a reversal only when perm is one.
          std::vector<int64_t> dims;
          for (int64_t p : val.perm)
            dims.push_back(tensorTy.getShape()[p]);

          auto transposedTy =
              mlir::RankedTensorType::get(dims, tensorTy.getElementType());
          auto destination =
              mlir::tensor::EmptyOp::create(builder, transposedTy, {});
          auto transpose = mlir::linalg::TransposeOp::create(
              builder, op, destination, val.perm);
          mlir::Value result = *transpose.result_begin();
          return result;
        },
        [&](const BasicOpReplicate &val) -> mlir::Value {
          auto op = LowerSubExp(val.val, ctx);
          auto elementTy = op.getType();

          // TODO dynamic replicate
          std::vector<int64_t> before;
          for (auto d : val.shape.dims)
            before.push_back(match(
                d,
                [&](const ConstantSubExp &) { return d.GetIntValue(); },
                [&](const VarSubExp &) -> int64_t { Undefined(); }));

          std::vector<int64_t> original;
          auto t = op.getType();
          if (auto tensorTy = llvm::dyn_cast<mlir::RankedTensorType>(t)) {
            auto dims = tensorTy.getShape();
            before.insert(before.end(), dims.begin(), dims.end());
            original.insert(original.end(), dims.begin(), dims.end());
            elementTy = tensorTy.getElementType();
          }

          else {
            auto rankedType = mlir::RankedTensorType::get({}, elementTy);
            op = mlir::tensor::FromElementsOp::create(builder, rankedType, op);
          }

          auto transposedTy = mlir::RankedTensorType::get(before, elementTy);
          auto destination =
              mlir::tensor::EmptyOp::create(builder, transposedTy, {});

          std::vector<int64_t> addedDims;
          for (int64_t i = 0; i < std::ssize(before) - std::ssize(original);
               i++) {
            addedDims.push_back(i);
          }

          auto broadcasted = mlir::linalg::BroadcastOp::create(
              builder, op, destination, addedDims);
          return *broadcasted.getResult().begin();
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
          auto cmpi = [&](mlir::arith::CmpIPredicate p) -> mlir::Value {
            return mlir::arith::CmpIOp::create(builder, p, lhs, rhs);
          };
          auto cmpf = [&](mlir::arith::CmpFPredicate p) -> mlir::Value {
            return mlir::arith::CmpFOp::create(builder, p, lhs, rhs);
          };
          using enum mlir::arith::CmpIPredicate;
          using enum mlir::arith::CmpFPredicate;
          return match(
              val.op,
              [&](const CmpOpEq &eqOp) -> mlir::Value {
                return match(
                    eqOp.t,
                    [&](const PrimTypeInt &) { return cmpi(eq); },
                    [&](const PrimTypeFloat &) -> mlir::Value {
                      return cmpf(OEQ);
                    },
                    [&](const PrimTypeBool &) -> mlir::Value { Undefined(); },
                    [&](const PrimTypeUnit &) -> mlir::Value { Undefined(); });
              },
              [&](const CmpOpUlt &) { return cmpi(ult); },
              [&](const CmpOpUle &) { return cmpi(ule); },
              [&](const CmpOpSlt &) { return cmpi(slt); },
              [&](const CmpOpSle &) { return cmpi(sle); },
              [&](const CmpOpFlt &) { return cmpf(OLT); },
              [&](const CmpOpFle &) { return cmpf(OLE); },
              [&](const CmpOpLLt &) -> mlir::Value { Undefined(); },
              [&](const CmpOpLLe &) -> mlir::Value { Undefined(); });
        },
        [](const BasicOpOpaque &) -> mlir::Value { Undefined(); },
        [](const BasicOpArrayVal &) -> mlir::Value { Undefined(); },
        [](const BasicOpUnOp &) -> mlir::Value { Undefined(); },
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
        [&](const GPUBody &) -> Values { Undefined(); });
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

  Values LowerSegMap(const SegLevel &lvl, const SegSpace &space,
                     const KernelBody &body, const std::vector<Type> &ret,
                     Ctx &ctx) {
    // Block-level and in-block kernels describe a nested iteration space that
    // a flat linalg.generic cannot express.
    RequireThreadLevel(lvl);

    // This lowers a SegMap to a linalg.generic op, whose
    //   * iteration space corresponds to the SegMap's SegSpace;
    //   * inputs correspond to "affine reads" in the SegMap body.
    // An affine read is an array load whose index expression is an affine
    // function of the iteration space.

    IterationSpace iterSpace = LowerSegSpace(space, ctx);

    std::vector<AffineRead> affine_reads =
        FindSegOpAffineReads(iterSpace, body);

    Values inputs;
    for (auto &read : affine_reads)
      inputs.push_back(ctx.subexps.lookup(read.array.name));

    auto shapeTy = toShapeType(std::views::values(space.dims));
    mlir::SmallVector<mlir::Type> returnTypes;
    for (const Type &resultType : ret) {
      if (auto *acc = GetAccType(resultType)) {
        assert(acc->ispace.dims.size() == 1);
        assert(acc->ts.size() == 1);
        auto updateType = LowerSegOpBaseType(acc->ts[0], ctx);
        returnTypes.push_back(mlir::RankedTensorType::get(shapeTy, updateType));
        returnTypes.push_back(mlir::RankedTensorType::get(
            shapeTy, builder.getI64Type()));
      } else {
        returnTypes.push_back(mlir::RankedTensorType::get(
            shapeTy, LowerSegOpBaseType(resultType, ctx)));
      }
    }
    Values dynamicSizes;
    for (const auto &d : iterSpace) {
      if (d.isDynamic)
        dynamicSizes.push_back(mlir::arith::IndexCastOp::create(
            builder, builder.getIndexType(), d.value));
    };
    Values outputs;
    for (auto &ty : returnTypes) {
      outputs.push_back(
          mlir::tensor::EmptyOp::create(builder, ty, dynamicSizes));
    }

    std::vector iterTypes(iterSpace.size(),
                          mlir::utils::IteratorType::parallel);

    // Map the iteration space to each input's dimensions.
    mlir::SmallVector<mlir::AffineMap> indexingMaps;
    for (const auto &read : affine_reads) {
      indexingMaps.push_back(read.indexMap);
    }

    // Map the iteration space to each output's dimensions.
    auto rank = iterSpace.size();
    indexingMaps.append(
        outputs.size(),
        mlir::AffineMap::getMultiDimIdentityMap(rank, &context));

    auto op = mlir::linalg::GenericOp::create(
        builder,
        returnTypes,
        inputs,
        outputs,
        indexingMaps,
        iterTypes,
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToEnd(b.getInsertionBlock());

          Ctx local = ctx;
          Values results = LowerSegOpKernelBody(
              loc, args, affine_reads, iterSpace, body, ret, local);

          assert(results.size() == outputs.size());
          mlir::linalg::YieldOp::create(builder, loc, results);
        });

    return op.getResults();
  }

  // TODO Same limitations as SegMap, probably.
  Values LowerSegRed(const SegRed &pSegRed, Ctx &ctx) {
    RequireThreadLevel(pSegRed.lvl);

    if (pSegRed.ops.size() != 1) {
      Undefined();
    }
    auto pSegBinOp = pSegRed.ops[0];
    if (pSegBinOp.shape.dims.size() != 0) {
      Undefined();
    }
    Values neutral;
    for (auto &e : pSegBinOp.neutral) {
      neutral.push_back(LowerSubExp(e, ctx));
    }

    IterationSpace iterSpace = LowerSegSpace(pSegRed.space, ctx);
    int64_t rank = std::ssize(iterSpace);
    if (rank == 0)
      Undefined();

    std::vector<AffineRead> affine_reads =
        FindSegOpAffineReads(iterSpace, pSegRed.body);

    Values inputs;
    for (auto &read : affine_reads)
      inputs.push_back(ctx.subexps.lookup(read.array.name));

    mlir::SmallVector<mlir::Type> returnTypes;
    for (auto ret : pSegBinOp.lambda.ret) {
      auto baseTy = LowerSegOpBaseType(ret, ctx);
      auto shapeTy = toShapeType(std::views::values(pSegRed.space.dims) |
                                 std::views::take(rank - 1));
      returnTypes.push_back(mlir::RankedTensorType::get(shapeTy, baseTy));
    }
    Values dynamicSizes;
    int64_t reductionIndex = rank - 1;
    for (const auto &d : iterSpace) {
      if (d.index == reductionIndex)
        continue;
      if (d.isDynamic)
        dynamicSizes.push_back(mlir::arith::IndexCastOp::create(
            builder, builder.getIndexType(), d.value));
    };
    Values outputs;
    for (auto [ty, ne] : llvm::zip_equal(returnTypes, neutral)) {
      outputs.push_back(mlir::linalg::FillOp::create(
                            builder,
                            ne,
                            mlir::ValueRange{mlir::tensor::EmptyOp::create(
                                builder, ty, dynamicSizes)})
                            .result());
    }

    std::vector iterTypes(iterSpace.size(),
                          mlir::utils::IteratorType::parallel);
    iterTypes[rank - 1] = mlir::utils::IteratorType::reduction;

    // Map the iteration space to each input's dimensions.
    mlir::SmallVector<mlir::AffineMap> indexingMaps;
    for (const auto &read : affine_reads) {
      indexingMaps.push_back(read.indexMap);
    }

    // Map the iteration space to each output's dimensions.
    mlir::SmallVector<mlir::AffineExpr> outDims;
    for (int64_t i = 0; i < rank - 1; ++i) {
      outDims.push_back(mlir::getAffineDimExpr(i, &context));
    }
    indexingMaps.append(outputs.size(),
                        mlir::AffineMap::get(rank, 0, outDims, &context));

    auto op = mlir::linalg::GenericOp::create(
        builder,
        returnTypes,
        inputs,
        outputs,
        indexingMaps,
        iterTypes,
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToEnd(b.getInsertionBlock());

          Ctx local = ctx;
          Values returns = LowerSegOpKernelBody(
              loc, args, affine_reads, iterSpace, pSegRed.body, pSegRed.ret, local);

          // Bind accumulators and inputs to reduce op.
          auto accs = args.drop_front(inputs.size());
          assert(pSegBinOp.lambda.params.size() ==
                 accs.size() + returns.size());
          for (auto [param, acc] : llvm::zip(pSegBinOp.lambda.params, accs)) {
            local.subexps.insert(param.name, acc);
          }
          for (auto [param, ret] : llvm::zip_equal(
                   llvm::drop_begin(pSegBinOp.lambda.params, accs.size()),
                   returns)) {
            local.subexps.insert(param.name, ret);
          }
          auto results = LowerBody(pSegBinOp.lambda.body, local);

          mlir::linalg::YieldOp::create(builder, loc, results);
        });

    return op.getResults();
  }

  Values LowerSegHist(const SegHist &pSegHist, Ctx &ctx) {
    RequireThreadLevel(pSegHist.lvl);

    // If this map is the identity function, IREE will kill it.
    Values inputs = LowerSegMap(
        pSegHist.lvl, pSegHist.space, pSegHist.body, pSegHist.ret, ctx);
    if (inputs.size() != 2)
      Undefined();
    mlir::Value indices = inputs[0];
    mlir::Value values = inputs[1];

    // Histogram
    if (pSegHist.ops.size() != 1) {
      Undefined();
    }
    auto pHistOp = pSegHist.ops[0];

    Values dests;
    for (auto const &vn : pHistOp.dest) {
      dests.push_back(LowerSubExp(vn.name, ctx));
    }
    if (dests.size() != 1)
      Undefined();
    auto dest = dests[0];

    if (getRankedType(indices.getType()).getRank() != 1)
      Undefined();

    auto op = mlir::iree_compiler::IREE::LinalgExt::ScatterOp::create(
        builder, dest.getType(), values, indices, {}, dest, {0}, false);

    // Lower the histogram's combining op.
    mlir::SmallVector<mlir::Type> opInputTypes;
    for (const auto &param : pHistOp.lambda.params)
      opInputTypes.push_back(LowerTy(param.dec.v));
    mlir::SmallVector<mlir::Location> locs;
    for (const auto &_ : opInputTypes)
      locs.push_back(builder.getLoc());
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto *blk = builder.createBlock(&op.getRegion(), {}, opInputTypes, locs);
      Ctx local = ctx;
      for (auto [param, arg] :
           llvm::zip_equal(pHistOp.lambda.params, blk->getArguments()))
        local.subexps.insert(param.name, arg);
      auto returns = LowerBody(pHistOp.lambda.body, local);
      mlir::iree_compiler::IREE::LinalgExt::YieldOp::create(builder, returns);
    }

    return {op.getResult(0)};
  }

  // TODOs:
  // Support vector SegBinOps
  // Support multiple SegBinOps
  // Support size(SegBinOp) > 0
  // Parallelise
  // XXX: There seems to be bug in IREE forall and IREE code result in segfault.
  // (#24775)
  Values LowerSegScan(const SegScan &pSegScan, Ctx &ctx) {
    RequireThreadLevel(pSegScan.lvl);

    if (pSegScan.ops.size() != 1)
      Undefined();

    const SegBinOp &segBinOp = pSegScan.ops[0];

    if (!segBinOp.shape.dims.empty())
      Undefined();

    const int64_t scanValueCount = segBinOp.neutral.size();
    const int64_t outputCount = pSegScan.ret.size();

    IterationSpace iterSpace = LowerSegSpace(pSegScan.space, ctx);
    if (iterSpace.size() == 0)
      Undefined();
    const Dim &scanDim = iterSpace.back();
    auto beforeLastDim = std::span{iterSpace}.first(iterSpace.size() - 1);

    mlir::Location loc = builder.getLoc();

    Values neutrals;
    neutrals.reserve(scanValueCount);
    for (const SubExp &neutral : segBinOp.neutral)
      neutrals.push_back(LowerSubExp(neutral, ctx));

    auto shape = toShapeType(std::views::values(pSegScan.space.dims));
    mlir::SmallVector<mlir::RankedTensorType> returnTypes;
    returnTypes.reserve(outputCount);
    for (const Type &ret : pSegScan.ret)
      returnTypes.push_back(
          mlir::RankedTensorType::get(shape, LowerSegOpBaseType(ret, ctx)));

    mlir::SmallVector<mlir::Value> dynamicSizes;
    for (auto &dim : iterSpace) {
      if (dim.isDynamic) {
        dynamicSizes.push_back(mlir::arith::IndexCastOp::create(
            builder, builder.getIndexType(), dim.value));
      }
    }

    Values initialOutputs;
    initialOutputs.reserve(outputCount);
    for (mlir::RankedTensorType returnType : returnTypes)
      initialOutputs.push_back(
          mlir::tensor::EmptyOp::create(builder, returnType, dynamicSizes));

    auto zero = mlir::arith::ConstantIndexOp::create(builder, 0);
    auto one = mlir::arith::ConstantIndexOp::create(builder, 1);

    mlir::Value scanUpperBound = scanDim.value;
    if (!scanUpperBound.getType().isIndex()) {
      scanUpperBound = mlir::arith::IndexCastOp::create(
          builder, builder.getIndexType(), scanUpperBound);
    }
    mlir::SmallVector<mlir::OpFoldResult> mapUpperBounds;
    for (auto &dim : beforeLastDim) {
      mlir::Value upperBound = dim.value;
      if (!upperBound.getType().isIndex()) {
        upperBound = mlir::arith::IndexCastOp::create(
            builder, loc, builder.getIndexType(), upperBound);
      }
      mapUpperBounds.push_back(upperBound);
    }

    mlir::SmallVector<mlir::RankedTensorType> rowTypes;
    rowTypes.reserve(outputCount);
    for (mlir::RankedTensorType returnType : returnTypes)
      rowTypes.push_back(mlir::RankedTensorType::get(
          {shape.back()}, returnType.getElementType()));

    mlir::SmallVector<mlir::Value> rowDynamicSizes;
    if (scanDim.isDynamic) {
      rowDynamicSizes.push_back(scanUpperBound);
    }

    auto forallOp = mlir::scf::ForallOp::create(
        builder,
        loc,
        mapUpperBounds,
        mlir::ValueRange{initialOutputs},
        /*mapping=*/std::nullopt,
        [&](mlir::OpBuilder &forallBuilder,
            mlir::Location forallLoc,
            mlir::ValueRange regionArgs) {
          mlir::OpBuilder::InsertionGuard outerGuard(builder);
          builder.setInsertionPoint(forallBuilder.getInsertionBlock(),
                                    forallBuilder.getInsertionPoint());

          Ctx local = ctx;

          int64_t mapRank = mapUpperBounds.size();
          mlir::ValueRange ivs = regionArgs.take_front(mapRank);
          mlir::ValueRange sharedOutputs = regionArgs.drop_front(mapRank);

          for (int64_t i = 0; i < std::ssize(ivs); i++) {
            const auto &dim = beforeLastDim[i];
            mlir::Value gtid = ivs[i];

            if (gtid.getType() != dim.value.getType()) {
              gtid = mlir::arith::IndexCastOp::create(
                  builder, loc, dim.value.getType(), gtid);
            }
            local.subexps.insert(dim.id, gtid);
          }

          Values initialRows;
          initialRows.reserve(outputCount);
          for (mlir::RankedTensorType rowType : rowTypes) {
            initialRows.push_back(mlir::tensor::EmptyOp::create(
                builder, forallLoc, rowType, rowDynamicSizes));
          }
          Values initialLoopValues = neutrals;
          initialLoopValues.append(initialRows.begin(), initialRows.end());

          auto scanLoop = mlir::scf::ForOp::create(
              builder,
              forallLoc,
              zero,
              scanUpperBound,
              one,
              initialLoopValues,

              [&](mlir::OpBuilder &bodyBuilder,
                  mlir::Location bodyLoc,
                  mlir::Value scanIndex,
                  mlir::ValueRange iterArgs) {
                mlir::OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPoint(bodyBuilder.getInsertionBlock(),
                                          bodyBuilder.getInsertionPoint());

                mlir::ValueRange accumulators =
                    iterArgs.take_front(scanValueCount);
                mlir::ValueRange outputRows =
                    iterArgs.drop_front(scanValueCount);

                Values shapedOutputRows(outputRows.begin(), outputRows.end());
                if (!rowDynamicSizes.empty()) {
                  for (mlir::Value &outputRow : shapedOutputRows) {
                    outputRow = mlir::iree_compiler::IREE::Flow::
                        TensorTieShapeOp::create(
                            builder, bodyLoc, outputRow, rowDynamicSizes);
                  }
                }

                mlir::Value gtid = scanIndex;
                if (gtid.getType() != scanDim.value.getType()) {
                  gtid = mlir::arith::IndexCastOp::create(
                      builder, bodyLoc, scanDim.value.getType(), scanIndex);
                }
                local.subexps.insert(scanDim.id, gtid);

                // Lower the kernel body
                for (const Stm &stm : pSegScan.body.stms)
                  LowerStm(stm, local);

                Values kernelResults;
                kernelResults.reserve(scanValueCount);
                for (const KernelResult &result : pSegScan.body.result) {
                  kernelResults.push_back(LowerSubExp(result.result, local));
                }

                assert(segBinOp.lambda.params.size() ==
                       accumulators.size() + kernelResults.size());
                Ctx lambdaCtx = local;
                for (auto [param, acc] :
                     llvm::zip(segBinOp.lambda.params, accumulators)) {
                  lambdaCtx.subexps.insert(param.name, acc);
                }
                for (auto [param, ret] :
                     llvm::zip_equal(llvm::drop_begin(segBinOp.lambda.params,
                                                      accumulators.size()),
                                     kernelResults)) {
                  lambdaCtx.subexps.insert(param.name, ret);
                }
                Values nextAccumulators =
                    LowerBody(segBinOp.lambda.body, lambdaCtx);

                Ctx postCtx = local;
                for (int64_t i = 0; i < scanValueCount; ++i) {
                  postCtx.subexps.insert(pSegScan.post_op.lambda.params[i].name,
                                         nextAccumulators[i]);
                }
                Values outputElements =
                    LowerBody(pSegScan.post_op.lambda.body, postCtx);

                Values nextRows;
                nextRows.reserve(outputCount);
                for (int64_t i = 0; i < outputCount; ++i) {
                  nextRows.push_back(mlir::tensor::InsertOp::create(
                      builder,
                      bodyLoc,
                      outputElements[i],
                      shapedOutputRows[i],
                      mlir::ValueRange{scanIndex}));
                }

                Values carry = nextAccumulators;
                carry.append(nextRows.begin(), nextRows.end());
                mlir::scf::YieldOp::create(builder, bodyLoc, carry);
              });

          mlir::SmallVector<mlir::OpFoldResult> offsets;
          mlir::SmallVector<mlir::OpFoldResult> sizes;
          mlir::SmallVector<mlir::OpFoldResult> strides;
          for (mlir::Value mapIndex : ivs) {
            offsets.push_back(mapIndex);
            sizes.push_back(builder.getIndexAttr(1));
            strides.push_back(builder.getIndexAttr(1));
          }
          offsets.push_back(builder.getIndexAttr(0));
          if (rowTypes.front().isDynamicDim(0)) {
            sizes.push_back(scanUpperBound);
          } else {
            sizes.push_back(
                builder.getIndexAttr(rowTypes.front().getDimSize(0)));
          }
          strides.push_back(builder.getIndexAttr(1));

          auto inParallel = mlir::scf::InParallelOp::create(builder, forallLoc);

          builder.setInsertionPointToStart(inParallel.getBody());

          auto completedRows = scanLoop.getResults().drop_front(scanValueCount);
          for (int64_t i = 0; i < outputCount; i++) {
            mlir::tensor::ParallelInsertSliceOp::create(builder,
                                                        forallLoc,
                                                        completedRows[i],
                                                        sharedOutputs[i],
                                                        offsets,
                                                        sizes,
                                                        strides);
          }
        });

    Values results;
    results.reserve(outputCount);
    for (mlir::Value result : forallOp.getResults()) {
      if (!dynamicSizes.empty()) {
        result = mlir::iree_compiler::IREE::Flow::TensorTieShapeOp::create(
            builder, loc, result, dynamicSizes);
      }
      results.push_back(result);
    }

    return results;
  }

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

  Values LowerSegOpKernelBody(mlir::Location loc,
                              const mlir::ValueRange blockArgs,
                              const std::vector<AffineRead> &affine_reads,
                              const IterationSpace &iterSpace,
                              const KernelBody &body,
                              const std::vector<Type> &resultTypes,
                              Ctx &ctx) {
    std::unordered_set<std::string> skipLowering;
    for (auto &read : affine_reads)
      skipLowering.insert(read.result.name);

    // Lower the kernel's affine reads by binding them to the inputs.
    for (auto [read, arg] : llvm::zip_equal(affine_reads, blockArgs)) {
      ctx.subexps.insert(read.result.name, arg);
    }
    // Lower the kernel's gtids by binding them to the iteration indices.
    // TODO only do this if they're actually used
    for (auto d : iterSpace) {
      auto idx = mlir::linalg::IndexOp::create(builder, loc, d.index);
      auto futharkIndexTy = d.value.getType();
      auto val =
          mlir::arith::IndexCastOp::create(builder, loc, futharkIndexTy, idx);
      ctx.subexps.insert(d.id, val);
    }
    // Lower the kernel body, skipping the affine reads.
    for (const Stm &stm : body.stms) {
      assert(stm.pat.elems.size() == 1);
      if (skipLowering.contains(stm.pat.elems[0].name.name))
        continue;
      LowerStm(stm, ctx);
    }
    Values returns;
    for (auto [result, type] : llvm::zip_equal(body.result, resultTypes)) {
      if (IsAccType(type)) {
        AccValue acc = ctx.accs.lookup(result.result.GetVName().name);
        returns.push_back(acc.update);
        returns.push_back(acc.index);
      } else {
        returns.push_back(LowerSubExp(result.result, ctx));
      }
    }

    return returns;
  }

  mlir::Value LowerPrimValue(PrimValue value, Ctx &ctx) {
    return mlir::arith::ConstantOp::create(builder, getPrimValueAttr(value));
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

  mlir::Value LowerBinOp(BinOp binOp, mlir::Value op0, mlir::Value op1,
                         Ctx &ctx) {
    auto isTensor = llvm::isa<mlir::TensorType>(op0.getType());
    if (auto *add = std::get_if<BinOpAdd>(&binOp.v)) {

      if (isTensor) {
        auto emptyTensor = mlir::tensor::EmptyOp::create(
            builder, op0.getType(), mlir::ValueRange{});
        return mlir::linalg::AddOp::create(builder, {op0, op1}, {emptyTensor})
            .getResult(0);
      }

      return mlir::arith::AddIOp::create(builder, {op0, op1}).getResult();
    }

    if (auto *mul = std::get_if<BinOpMul>(&binOp.v)) {

      if (isTensor) {
        auto emptyTensor = mlir::tensor::EmptyOp::create(
            builder, op0.getType(), mlir::ValueRange{});
        return mlir::linalg::MulOp::create(builder, {op0, op1}, {emptyTensor})
            .getResult(0);
      }

      return mlir::arith::MulIOp::create(builder, {op0, op1}).getResult();
    }

    if (std::get_if<BinOpSub>(&binOp.v)) {
      assert(!isTensor);

      return mlir::arith::SubIOp::create(builder, {op0, op1}).getResult();
    }

    // Futhark's `squot` truncates towards zero, which is what arith.divsi
    // does; `sdiv`, which rounds towards negative infinity, is a different op.
    if (std::get_if<BinOpSQuot>(&binOp.v)) {
      assert(!isTensor);

      return mlir::arith::DivSIOp::create(builder, {op0, op1}).getResult();
    }

    if (auto *div = std::get_if<BinOpSDivUp>(&binOp.v)) {
      assert(!isTensor);

      return mlir::arith::CeilDivSIOp::create(builder, {op0, op1}).getResult();
    }

    if (auto *fadd = std::get_if<BinOpFAdd>(&binOp.v)) {
      assert(!isTensor);

      return mlir::arith::AddFOp::create(builder, {op0, op1}).getResult();
    }

    if (auto *fmul = std::get_if<BinOpFMul>(&binOp.v)) {
      assert(!isTensor);

      return mlir::arith::MulFOp::create(builder, {op0, op1}).getResult();
    }

    if (auto *fsub = std::get_if<BinOpFSub>(&binOp.v)) {
      assert(!isTensor);

      return mlir::arith::SubFOp::create(builder, {op0, op1}).getResult();
    }

    if (auto *fdiv = std::get_if<BinOpFDiv>(&binOp.v)) {
      assert(!isTensor);

      return mlir::arith::DivFOp::create(builder, {op0, op1}).getResult();
    }

    // Futhark's `fmax` follows libm in returning the non-NaN operand, which is
    // arith.maxnumf rather than arith.maximumf.
    if (auto *fmax = std::get_if<BinOpFMax>(&binOp.v)) {
      assert(!isTensor);

      return mlir::arith::MaxNumFOp::create(builder, {op0, op1}).getResult();
    }

    if (auto *fpow = std::get_if<BinOpFPow>(&binOp.v)) {
      assert(!isTensor);

      return mlir::math::PowFOp::create(builder, op0, op1).getResult();
    }

    Undefined();
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
      return mlir::arith::ConstantOp::create(
          builder,
          mlir::DenseElementsAttr::get(getRankedType(tensorTy), attrs));
    } else {
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

  Values LowerIntrinsic(const std::string &fname, mlir::ValueRange args) {
    // `cond_t` is Futhark's branchless select; its suffix is a full type name
    // ("cond_i64", "cond_bool"), not the width suffix IntrinsicBaseName knows.
    if (fname.starts_with("cond_")) {
      if (args.size() != 3)
        Undefined();
      return {mlir::arith::SelectOp::create(builder, args[0], args[1], args[2])
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

    using namespace mlir::math;
    const auto &n = *base;
    if (n == "sqrt")
      return unary.operator()<SqrtOp>();
    if (n == "rsqrt")
      return unary.operator()<RsqrtOp>();
    if (n == "cbrt")
      return unary.operator()<CbrtOp>();
    if (n == "exp")
      return unary.operator()<ExpOp>();
    if (n == "exp2")
      return unary.operator()<Exp2Op>();
    if (n == "expm1")
      return unary.operator()<ExpM1Op>();
    if (n == "log")
      return unary.operator()<LogOp>();
    if (n == "log2")
      return unary.operator()<Log2Op>();
    if (n == "log10")
      return unary.operator()<Log10Op>();
    if (n == "log1p")
      return unary.operator()<Log1pOp>();
    if (n == "sin")
      return unary.operator()<SinOp>();
    if (n == "cos")
      return unary.operator()<CosOp>();
    if (n == "tan")
      return unary.operator()<TanOp>();
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
    if (n == "tanh")
      return unary.operator()<TanhOp>();
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
    if (n == "ceil")
      return unary.operator()<CeilOp>();
    if (n == "floor")
      return unary.operator()<FloorOp>();
    if (n == "trunc")
      return unary.operator()<TruncOp>();
    // Futhark's `round` is C's `rint` under the default rounding mode, i.e.
    // ties go to even.
    if (n == "round")
      return unary.operator()<RoundEvenOp>();
    if (n == "isnan")
      return unary.operator()<IsNaNOp>();
    if (n == "isinf")
      return unary.operator()<IsInfOp>();
    if (n == "popc")
      return unary.operator()<CtPopOp>();
    if (n == "clz")
      return unary.operator()<CountLeadingZerosOp>();
    if (n == "ctz")
      return unary.operator()<CountTrailingZerosOp>();
    if (n == "atan2")
      return binary.operator()<Atan2Op>();
    if (n == "copysign")
      return binary.operator()<CopySignOp>();
    if (n == "fma" || n == "mad") {
      if (args.size() != 3)
        Undefined();
      return {FmaOp::create(builder, args[0], args[1], args[2]).getResult()};
    }

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
      return LowerPrimType(val->t);
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
