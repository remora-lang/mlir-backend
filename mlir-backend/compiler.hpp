#pragma once
// Represents the state of a Futhark function
#include "blackbox.hpp"
#include "debug.hpp"
#include "ir/core.hpp"
#include "ir/segop.hpp"
#include "ir/syntax.hpp"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "utils.hpp"
#include "llvm/Support/ErrorHandling.h"
#include <format>
#include <iterator>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
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

struct Ctx {
  Env<std::string, mlir::Value> subexps;
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
    context.getOrLoadDialect<mlir::iree_compiler::IREE::Flow::FlowDialect>();
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

    std::vector<mlir::Type> inputTypes;
    for (const auto &param : fun.params) {
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
      for (auto [i, param] : llvm::enumerate(fun.params)) {
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

    for (auto [param, arg] : llvm::zip_equal(fun.params, func.getArguments())) {
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
    for (auto [elem, value] : llvm::zip_equal(stm.pat.elems, vs)) {
      ctx.subexps.insert(elem.name.name, value);
    }
  }

  Values LowerExp(const Exp &exp, Ctx &ctx) {
    return match(
        exp,
        [&](const ExpBasicOp &e) { return Values{LowerBasicOp(e.op, ctx)}; },
        [&](const ExpSubExp &e) { return Values{LowerSubExp(e.subExp, ctx)}; },
        [&](const ExpHostOp &e) { return LowerHostOp(e.op, ctx); },
        [&](const ExpApply &e) -> Values {
          auto fun = GetFunction(prog, e.fname);
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
                }
                llvm_unreachable("LowerBlackBox");
              });

          Values results;
          for (auto [v, ty] : llvm::zip_equal(returns, retTypes)) {
            results.push_back(castToType(v, ty));
          }
          return results;
        },
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
        });
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

          auto op1 = LowerSubExp(val.total, ctx);
          operands.push_back(op1);

          auto concat = mlir::tensor::ConcatOp::create(builder, 0, operands);
          return concat->getResult(0);
        },
        [&](const BasicOpFlatIndex &val) -> mlir::Value {
          auto tensor = LowerSubExp(val.base, ctx);
          std::vector<mlir::Value> indices;
          for (auto idx : val.operands) {
            auto i = LowerSubExp(idx, ctx);
            auto casted = mlir::arith::IndexCastOp::create(
                builder, builder.getIndexType(), i);
            indices.push_back(casted);
          }

          auto result =
              mlir::tensor::ExtractOp::create(builder, tensor, indices);
          return result.getResult();
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
              [](const auto &) -> mlir::Value { Undefined(); });
        },
        [&](const BasicOpReshape &val) -> mlir::Value {
          auto op0 = LowerSubExp(val.op0, ctx);
          std::vector<mlir::ReassociationIndices> reassociations;

          mlir::ReassociationIndices ranges;
          for (auto i = 0; i < val.dimEnd; i++)
            ranges.push_back(i);
          reassociations.push_back(ranges);

          auto prevTy = llvm::dyn_cast<mlir::RankedTensorType>(op0.getType());
          if (!prevTy)
            Undefined();
          for (int64_t i = val.dimEnd; i < std::ssize(prevTy.getShape()); i++) {
            reassociations.push_back({i});
          }

          std::vector<int64_t> dims;
          for (auto d : val.remainder.dims)
            dims.push_back(d.GetIntValue());

          auto resultTy =
              mlir::RankedTensorType::get(dims, prevTy.getElementType());
          auto reshape = mlir::tensor::CollapseShapeOp::create(
              builder, resultTy, op0, reassociations);
          return reshape->getResult(0);
        },
        [&](const BasicOpRearrange &val) -> mlir::Value {
          auto op = LowerSubExp(val.arr.name, ctx);
          auto tensorTy = llvm::dyn_cast<mlir::RankedTensorType>(op.getType());
          if (!tensorTy)
            Undefined();

          std::vector<int64_t> dims = tensorTy.getShape();
          std::reverse(dims.begin(), dims.end());

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
        [](const BasicOpIndex &) -> mlir::Value { Undefined(); },
        [](const BasicOpUpdate &) -> mlir::Value { Undefined(); },
        [](const BasicOpAssert &) -> mlir::Value { Undefined(); },
        [](const BasicOpFlatUpdate &) -> mlir::Value { Undefined(); },
        [](const BasicOpManifest &) -> mlir::Value { Undefined(); });
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
        [&](const SegScan &v) { return LowerSegScan(v, ctx); });
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
    // This lowers a SegMap to a linalg.generic op, whose
    //   * iteration space corresponds to the SegMap's SegSpace;
    //   * inputs correspond to "affine reads" in the SegMap body.
    // An affine read is an array load whose index expression is an affine
    // function of the iteration space.

    IterationSpace iterSpace = LowerSegSpace(pSegMap.space, ctx);

    std::vector<AffineRead> affine_reads =
        FindSegOpAffineReads(iterSpace, pSegMap.body);

    Values inputs;
    for (auto &read : affine_reads)
      inputs.push_back(ctx.subexps.lookup(read.array.name));

    mlir::SmallVector<mlir::Type> returnTypes;
    for (auto ret : pSegMap.ret) {
      auto baseTy = LowerSegOpBaseType(ret, ctx);
      auto shapeTy = toShapeType(std::views::values(pSegMap.space.dims));
      returnTypes.push_back(mlir::RankedTensorType::get(shapeTy, baseTy));
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
              loc, args, affine_reads, iterSpace, pSegMap.body, local);

          assert(results.size() == outputs.size());
          mlir::linalg::YieldOp::create(builder, loc, results);
        });

    return op.getResults();
  }

  // TODO Same limitations as SegMap, probably.
  Values LowerSegRed(const SegRed &pSegRed, Ctx &ctx) {
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
              loc, args, affine_reads, iterSpace, pSegRed.body, local);

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

  // TODOs:
  // Support vector SegBinOps
  // Support multiple SegBinOps
  // Support size(SegBinOp) > 0
  // Parallelise
  // XXX: There seems to be bug in IREE forall and IREE code result in segfault.
  // (#24775)
  Values LowerSegScan(const SegScan &pSegScan, Ctx &ctx) {

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
      if (auto idx = std::get_if<BasicOpFlatIndex>(&e->op.v)) {
        auto vnArray = idx->base.GetVName();

        std::vector<mlir::AffineExpr> usedDims;
        for (auto operand : idx->operands) {

          auto vnDim = operand.GetVName();
          auto d = std::ranges::find(iterSpace, vnDim.name, &Dim::id);
          if (d == iterSpace.end())
            // TODO suport affine indexing beyond seg space ids
            Undefined();
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
                              const KernelBody &body, Ctx &ctx) {
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
    for (auto &r : body.result)
      returns.push_back(LowerSubExp(r.result, ctx));

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
      Undefined();
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

  static FunDef GetFunction(Prog &prog, std::string fname) {
    for (auto f : prog.funs) {
      if (f.name == fname)
        return f;
    }

    Undefined();
  }
};
