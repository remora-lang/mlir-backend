#pragma once
// Represents the state of a Futhark function
struct Ctx {
  std::unordered_map<std::string, mlir::Value> subexps;
};

struct FutharkCompiler {
  Prog &prog;

  std::unordered_map<FunDef, mlir::func::FuncOp, FunDefHasher> functions;

  mlir::MLIRContext &context;

  mlir::ImplicitLocOpBuilder &builder;

  mlir::ModuleOp module;

  FutharkCompiler(Prog &prog, mlir::MLIRContext &context, mlir::ImplicitLocOpBuilder &builder)
      : prog(prog), context(context), builder(builder) {
    context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
    context.getOrLoadDialect<mlir::tensor::TensorDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::index::IndexDialect>();
    context.getOrLoadDialect<mlir::BuiltinDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
  }

  mlir::func::FuncOp LowerFunction(FunDef fun) {
    if (functions.find(fun) != functions.end())
      return functions[fun];

    auto lastPos = builder.saveInsertionPoint();
    builder.setInsertionPointToEnd(module.getBody());

    std::vector<mlir::Type> retTypes;
    std::vector<mlir::Type> inputTypes;
    for (auto t : fun.retType) {
      retTypes.push_back(LowerTy(t.first.v));
    }

    for (auto t : fun.params) {
      inputTypes.push_back(LowerTy(t.dec.v));
    }

    auto fType = builder.getFunctionType(inputTypes, retTypes);
    auto func = mlir::func::FuncOp::create(builder.getUnknownLoc(), fun.name, fType);
    builder.insert(func);
    functions[fun] = func;

    Ctx ctx;
    auto entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    for (auto i = 0; i < fun.params.size(); i++) {
      ctx.subexps[fun.params[i].name] = func.getArgument(i);
    }

    auto retValue = LowerBody(fun.body, ctx);
    mlir::func::ReturnOp::create(builder, builder.getUnknownLoc(), {retValue});
    builder.restoreInsertionPoint(lastPos);
    return func;
  }

  mlir::Value LowerBody(Body body, Ctx &ctx) {
    for (auto stm : body.stms) {
      LowerStm(stm, ctx);
    }

    return LowerSubExp(body.result[0], ctx);
  }

  void LowerStm(Stm stm, Ctx &ctx) {
    auto v = LowerExp(stm.exp, ctx);
    assert(stm.pat.elems.size() == 1);
    ctx.subexps[stm.pat.elems[0].name.name] = v;
  }

  mlir::Value LowerExp(Exp exp, Ctx &ctx) {
    if (auto *val = std::get_if<ExpBasicOp>(&exp.v)) {
      return LowerBasicOp(val->op, ctx);
    }

    if (auto *val = std::get_if<ExpSubExp>(&exp.v)) {
      return LowerSubExp(val->subExp, ctx);
    }

    if (auto *val = std::get_if<ExpSoacOp>(&exp.v)) {
      return LowerSoac(val->soac, ctx);
    }

    if (auto *val = std::get_if<ExpApply>(&exp.v)) {
      auto func = GetFunction(prog, val->fname);
      auto llvmFunc = LowerFunction(func);

      std::vector<mlir::Value> args;
      for (auto i = 0; i < func.params.size(); i++) {
        auto [subExp, diet] = val->args[i];
        args.push_back(LowerSubExp(subExp, ctx));
      }

      auto call = mlir::func::CallOp::create(builder, llvmFunc, args);
      return call->getResult(0);
    }

    if (auto *val = std::get_if<ExpBasicOp>(&exp.v)) {
      return LowerBasicOp(val->op, ctx);
    }

    Unreachable();
  }

  mlir::Value LowerBasicOp(BasicOp basicOp, Ctx &ctx) {
    if (auto *val = std::get_if<BasicOpSubExp>(&basicOp.v)) {
      return LowerSubExp(val->subExp, ctx);
    }

    if (auto *val = std::get_if<BasicOpBinOp>(&basicOp.v)) {
      auto op0 = LowerSubExp(val->op0, ctx);
      auto op1 = LowerSubExp(val->op1, ctx);
      return LowerBinOp(val->op, op0, op1, ctx);
    }

    if (auto *val = std::get_if<BasicOpArrayLit>(&basicOp.v)) {
      return LowerArrayLit(*val, ctx);
    }

    if (auto *val = std::get_if<BasicOpIota>(&basicOp.v)) {
      auto n = (uint64_t)val->n.GetIntValue();
      uint64_t s = val->s.GetIntValue();
      auto iota = Iota(n, (uint64_t)val->x.GetIntValue(), s);

      // Get an array type for the iota
      auto primType = PrimTypeInt{val->t};
      Shape shape{};
      shape.dims.push_back(val->n);
      auto arrTy = Type::CreateArr(PrimType{primType}, shape);

      BasicOpArrayLit arrLit{};
      arrLit.t = arrTy;

      for (auto v : iota) {
        ConstantSubExp exp = {PrimValue::CreateInt((int64_t)val, (uint64_t)primType.t)};
        arrLit.values.push_back({exp});
      }

      return LowerArrayLit(arrLit, ctx);
    }

    if (auto *val = std::get_if<BasicOpConcat>(&basicOp.v)) {
      std::vector<mlir::Value> operands;
      for (auto arr : val->arrs)
        operands.push_back(LowerSubExp(arr.name, ctx));

      auto op1 = LowerSubExp(val->total, ctx);
      operands.push_back(op1);

      auto concat = mlir::tensor::ConcatOp::create(builder, 0, operands);
      return concat->getResult(0);
    }

    if (auto *val = std::get_if<BasicOpFlatIndex>(&basicOp.v)) {
      auto tensor = LowerSubExp(val->base, ctx);
      std::vector<mlir::Value> indices;
      for (auto idx : val->operands) {
        auto i = LowerSubExp(idx, ctx);
        auto casted = mlir::index::CastSOp::create(builder, builder.getIndexType(), i);
        indices.push_back(casted);
      }

      auto result = mlir::tensor::ExtractOp::create(builder, tensor, indices);
      return result.getResult();
    }

    if (auto *val = std::get_if<BasicOpConvOp>(&basicOp.v)) {
      auto op0 = LowerSubExp(val->op0, ctx);
      auto convOp = val->op;
      if (auto *zext = std::get_if<ConvOpZExt>(&convOp.v)) {
        auto primTy = Type::CreatePrim(PrimType::Int(zext->to));
        return mlir::arith::ExtUIOp::create(builder, LowerTy(primTy), op0);
      }

      if (auto *sext = std::get_if<ConvOpSExt>(&convOp.v)) {
        auto primTy = Type::CreatePrim(PrimType::Int(sext->to));
        return mlir::arith::ExtUIOp::create(builder, LowerTy(primTy), op0);
      }
    }

    if (auto *val = std::get_if<BasicOpReshape>(&basicOp.v)) {
      auto op0 = LowerSubExp(val->op0, ctx);
      std::vector<mlir::ReassociationIndices> reassociations;

      mlir::ReassociationIndices ranges;
      for (auto i = 0; i < val->dimEnd; i++)
        ranges.push_back(i);
      reassociations.push_back(ranges);

      auto prevTy = llvm::dyn_cast<mlir::RankedTensorType>(op0.getType());
      for (auto i = val->dimEnd; i < prevTy.getShape().size(); i++) {
        reassociations.push_back({i});
      }

      std::vector<int64_t> dims;
      for (auto d : val->remainder.dims)
        dims.push_back(d.GetIntValue());

      auto resultTy = mlir::RankedTensorType::get(dims, prevTy.getElementType());
      auto reshape = mlir::tensor::CollapseShapeOp::create(builder, resultTy, op0, reassociations);
      return reshape->getResult(0);
    }

    if (auto *val = std::get_if<BasicOpRearrange>(&basicOp.v)) {
      auto op = LowerSubExp(val->arr.name, ctx);
      auto tensorTy = llvm::dyn_cast<mlir::RankedTensorType>(op.getType());

      std::vector<int64_t> dims = tensorTy.getShape();
      std::reverse(dims.begin(), dims.end());

      auto transposedTy = mlir::RankedTensorType::get(dims, tensorTy.getElementType());
      auto destination = mlir::tensor::EmptyOp::create(builder, transposedTy, {});
      auto transpose = mlir::linalg::TransposeOp::create(builder, op, destination, val->perm);
      mlir::Value result = *transpose.result_begin();
      return result;
    }

    if (auto *val = std::get_if<BasicOpReplicate>(&basicOp.v)) {
      auto op = LowerSubExp(val->val, ctx);
      auto elementTy = op.getType();

      std::vector<int64_t> before;
      for (auto d : val->shape.dims)
        before.push_back(d.GetIntValue());

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
      auto destination = mlir::tensor::EmptyOp::create(builder, transposedTy, {});

      std::vector<int64_t> addedDims;
      for (auto i = 0; i < (before.size() - original.size()); i++) {
        addedDims.push_back(i);
      }

      auto broadcasted = mlir::linalg::BroadcastOp::create(builder, op, destination, addedDims);
      return *broadcasted.getResult().begin();
    }

    Unreachable();
  }

  mlir::Value LowerSubExp(SubExp subExp, Ctx &ctx) {
    if (auto *val = std::get_if<VarSubExp>(&subExp.v)) {
      return ctx.subexps[val->v.name];
    }

    if (auto *val = std::get_if<ConstantSubExp>(&subExp.v)) {
      return LowerPrimValue(val->v, ctx);
    }

    Unreachable();
  }

  mlir::Value LowerSubExp(std::string vname, Ctx &ctx) { return ctx.subexps[vname]; }

  mlir::Value LowerSoac(std::shared_ptr<Soac> pSoac, Ctx &ctx) {
    if (auto *val = std::get_if<SoacScrema>(&pSoac->v)) {
      return LowerScrema(*val, ctx);
    }

    Unreachable();
  }

  mlir::Value LowerScrema(SoacScrema screma, Ctx &ctx) {
    // The size of the input arrays
    auto inputSize = LowerSubExp(screma.w, ctx);

    // Lower the input arrs
    std::vector<mlir::Value> arrs;
    for (auto arr : screma.arrs) {
      arrs.push_back(LowerSubExp(arr.name, ctx));
    }

    if (screma.form.scremaLambda.ret.size() != 1) {
      throw std::runtime_error("TODO: Multiple return types");
    }

    // Compute the dimensions of the return type
    mlir::Type baseTy;
    std::vector<int64_t> dimensions;
    dimensions.push_back(screma.w.GetIntValue());
    if (auto *val = std::get_if<TypeArray<Shape, NoUniqueness>>(&screma.form.scremaLambda.ret[0].t.v)) {
      baseTy = LowerPrimType(val->elem);
      for (auto d : val->shape.dims)
        dimensions.push_back(d.GetIntValue());
    }

    else if (auto *val = std::get_if<TypePrim<Shape, NoUniqueness>>(&screma.form.scremaLambda.ret[0].t.v)) {
      baseTy = LowerPrimType(val->t);
    }

    else {
      throw std::runtime_error("Unsupported screma return type");
    }

    auto rvalueTy = mlir::RankedTensorType::get(dimensions, baseTy);

    mlir::Value carry = mlir::tensor::EmptyOp::create(builder, rvalueTy, {});

    auto idxTy = mlir::IntegerType::get(&context, 64);

    auto getIdx = [this](unsigned long long x) { return mlir::arith::ConstantIndexOp::create(builder, x).getResult(); };

    auto begin = getIdx(0);
    auto end = getIdx(screma.w.GetIntValue());
    auto step = getIdx(1);

    auto forOp = mlir::scf::ForOp::create(builder, begin, end, step, carry);
    auto indVar = forOp.getInductionVar();
    auto block = forOp.getBody();

    auto outputDest = carry;
    carry = block->getArgument(1);

    auto before = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(block);

    auto zero = getIdx(0);
    auto one = getIdx(1);

    Ctx scremaCtx = ctx;
    for (auto varIdx = 0; varIdx < screma.form.scremaLambda.params.size(); varIdx++) {
      auto value = ctx.subexps[screma.arrs[varIdx].name];

      auto t = value.getType();

      auto dd = llvm::dyn_cast<mlir::RankedTensorType>(t).getShape();

      std::vector<mlir::Value> _offsets;
      _offsets.push_back(indVar);
      for (auto i = 1; i < dd.size(); i++)
        _offsets.push_back(zero);

      std::vector<mlir::Value> _sizes;
      _sizes.push_back(getIdx(1));
      for (auto i = 1; i < dd.size(); i++)
        _sizes.push_back(getIdx(dd[i]));

      std::vector<mlir::Value> strides2;
      for (auto i = 0; i < dd.size(); i++)
        strides2.push_back(one);

      mlir::Value slice{};
      if (dd.size() <= 1) {
        slice = mlir::tensor::ExtractOp::create(builder, value, indVar).getResult();
      }

      else {
        std::vector<std::vector<mlir::OpFoldResult>> reassociation = {};

        for (auto vec : {_offsets, _sizes, strides2}) {

          std::vector<mlir::OpFoldResult> indices = {};
          for (auto elem : vec) {
            indices.push_back(mlir::getAsOpFoldResult(elem));
          }
          reassociation.push_back(indices);
        }

        slice =
            mlir::tensor::ExtractSliceOp::create(builder, value, reassociation[0], reassociation[1], reassociation[2])
                .getResult();

        std::vector<mlir::ReassociationIndices> toCollapse;
        toCollapse.push_back({0, 1});
        for (uint64_t i = 2; i < _offsets.size(); i++) {
          mlir::ReassociationIndices indices{};
          indices.push_back(i);
          toCollapse.push_back(indices);
        }

        auto currTy = llvm::dyn_cast<mlir::RankedTensorType>(slice.getType());
        std::vector<int64_t> dims = currTy.getShape();
        dims.erase(dims.begin());

        auto targetTy = mlir::RankedTensorType::get(dims, currTy.getElementType());

        auto reshaped = mlir::tensor::CollapseShapeOp::create(builder, targetTy, slice, toCollapse);
        slice = reshaped;
      }

      scremaCtx.subexps[screma.form.scremaLambda.params[varIdx].name] = slice;
    }

    auto execution = LowerBody(screma.form.scremaLambda.body, scremaCtx);

    std::vector<mlir::Value> offsets;
    offsets.push_back(indVar);
    for (auto i = 1; i < dimensions.size(); i++)
      offsets.push_back(zero);

    std::vector<mlir::Value> sizes;
    sizes.push_back(getIdx(1));
    for (auto i = 1; i < dimensions.size(); i++)
      sizes.push_back(getIdx(dimensions[i]));

    std::vector<mlir::Value> strides;
    for (auto i = 0; i < dimensions.size(); i++)
      strides.push_back(one);
    ;
    mlir::Value insert{};
    if (dimensions.size() <= 1) {
      insert = mlir::tensor::InsertOp::create(builder, execution, carry, {indVar}).getResult();
    }

    else {
      std::vector<std::vector<mlir::OpFoldResult>> reassociation = {};

      for (auto vec : {offsets, sizes, strides}) {

        std::vector<mlir::OpFoldResult> indices = {};
        for (auto elem : vec) {
          indices.push_back(mlir::getAsOpFoldResult(elem));
        }
        reassociation.push_back(indices);
      }

      insert = mlir::tensor::InsertSliceOp::create(
          builder, execution, carry, llvm::ArrayRef<mlir::OpFoldResult>(reassociation[0]),
          llvm::ArrayRef<mlir::OpFoldResult>(reassociation[1]), llvm::ArrayRef<mlir::OpFoldResult>(reassociation[2]));
    }

    mlir::scf::YieldOp::create(builder, insert);
    builder.restoreInsertionPoint(before);

    outputDest = forOp.getResult(0);
    if (screma.form.scremaReduces.size() == 0)
      return outputDest;

    auto red = LowerRedomap(screma, outputDest, ctx);
    return red;
  }

  mlir::Value LowerRedomap(SoacScrema screma, mlir::Value src, Ctx &ctx) {
    auto inputSize = LowerSubExp(screma.w, ctx);

    // Clone the context
    Ctx scremaCtx = ctx;

    auto inputTy = llvm::dyn_cast<mlir::RankedTensorType>(src.getType());
    mlir::RankedTensorType temp = mlir::RankedTensorType::Builder(inputTy).dropDim(0);
    mlir::Type rvalueTy = temp;
    if (temp.getShape().size() == 0)
      rvalueTy = temp.getElementType();

    auto red = screma.form.scremaReduces[0];
    assert(screma.form.scremaReduces.size() == 1);
    auto carry = LowerSubExp(red.neutral[0], ctx);
    assert(red.neutral.size() == 1);

    auto getIdx = [this](unsigned long long x) { return mlir::arith::ConstantIndexOp::create(builder, x).getResult(); };

    auto begin = getIdx(0);
    auto end = getIdx(screma.w.GetIntValue());
    auto step = getIdx(1);

    auto forOp = mlir::scf::ForOp::create(builder, begin, end, step, carry);
    auto indVar = forOp.getInductionVar();
    auto block = forOp.getBody();

    auto outputDest = carry;
    carry = block->getArgument(1);

    auto before = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(block);

    auto zero = getIdx(0);
    auto one = getIdx(1);

    scremaCtx.subexps[red.lambda.params[0].name] = carry;
    auto slice = mlir::tensor::ExtractOp::create(builder, src, indVar);
    scremaCtx.subexps[red.lambda.params[1].name] = slice;

    auto execution = LowerBody(red.lambda.body, scremaCtx);

    mlir::scf::YieldOp::create(builder, carry);
    builder.restoreInsertionPoint(before);

    return forOp.getResult(0);
  }

  mlir::Value LowerPrimValue(PrimValue value, Ctx &ctx) {
    auto v = value.v;
    if (auto *val = std::get_if<IntValue>(&v)) {
      auto t = builder.getIntegerType(GetWidth(*val));
      return mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(),
                                             builder.getIntegerAttr(t, GetValue(*val)));
    }

    if (auto *val = std::get_if<FloatValue>(&v)) {
      auto t = GetFloatType(builder, GetWidth(*val));
      return mlir::arith::ConstantOp::create(builder, builder.getFloatAttr(t, GetValue(*val)));
    }

    Unreachable();
  }

  mlir::Value LowerBinOp(BinOp binOp, mlir::Value op0, mlir::Value op1, Ctx &ctx) {
    auto isTensor = llvm::isa<mlir::TensorType>(op0.getType());
    if (auto *add = std::get_if<BinOpAdd>(&binOp.v)) {

      if (isTensor) {
        auto emptyTensor = mlir::tensor::EmptyOp::create(builder, op0.getType(), mlir::ValueRange{});
        return mlir::linalg::AddOp::create(builder, {op0, op1}, {emptyTensor}).getResult(0);
      }

      return mlir::arith::AddIOp::create(builder, {op0, op1}).getResult();
    }

    if (auto *mul = std::get_if<BinOpAdd>(&binOp.v)) {

      if (isTensor) {
        auto emptyTensor = mlir::tensor::EmptyOp::create(builder, op0.getType(), mlir::ValueRange{});
        return mlir::linalg::MulOp::create(builder, {op0, op1}, {emptyTensor}).getResult(0);
      }

      return mlir::arith::MulIOp::create(builder, {op0, op1}).getResult();
    }

    if (auto *fadd = std::get_if<BinOpFAdd>(&binOp.v)) {
      assert(!isTensor);

      return mlir::arith::AddFOp::create(builder, {op0, op1}).getResult();
    }

    if (auto *fmul = std::get_if<BinOpFMul>(&binOp.v)) {
      assert(!isTensor);

      return mlir::arith::MulFOp::create(builder, {op0, op1}).getResult();
    }
    
    if (auto *fdiv = std::get_if<BinOpFDiv>(&binOp.v)) {
      assert(!isTensor);

      return mlir::arith::DivFOp::create(builder, {op0, op1}).getResult();
    }

    if (auto *smod = std::get_if<BinOpSMod>(&binOp.v)) {
      assert(!isTensor);

      return mlir::arith::RemFOp::create(builder, {op0, op1}).getResult();
    }
        
    if (auto *srem = std::get_if<BinOpSRem>(&binOp.v)) {
      assert(!isTensor);

      return mlir::arith::RemSIOp::create(builder, {op0, op1}).getResult();
    }

    Unreachable();
  }

  mlir::Value LowerArrayLit(BasicOpArrayLit arrayLit, Ctx &ctx) {
    auto tensorTy = LowerTy(arrayLit.t);
    auto values = std::vector<mlir::Value>();
    for (auto subExp : arrayLit.values)
      values.push_back(LowerSubExp(subExp, ctx));
    auto tensor = mlir::tensor::FromElementsOp::create(builder, tensorTy, values);
    return tensor;
  }

  template <typename T> static std::vector<T> Iota(size_t n, T x, T s) {
    std::vector<T> elems;
    if (n <= 0)
      return elems;

    auto currentValue = x;
    for (auto i = 0; i < n; i++) {
      elems.push_back(currentValue);
      currentValue += s;
    }

    return elems;
  }

  mlir::Type LowerTy(Type t) {
    auto baseTy = &t.t.v;
    if (auto *val = std::get_if<TypePrim<Shape, NoUniqueness>>(baseTy)) {
      return LowerPrimType(val->t);
    }

    if (auto *val = std::get_if<TypeArray<Shape, NoUniqueness>>(baseTy)) {
      return LowerTypeArray(*val);
    }

    Unreachable();
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

    Unreachable();
  }

  mlir::Type LowerPrimTypeInt(PrimTypeInt primTypeInt) { return builder.getIntegerType(GetWidth(primTypeInt.t)); }

  mlir::Type LowerPrimTypeFloat(PrimTypeFloat primTypeFloat) {
    auto w = GetWidth(primTypeFloat.t);
    if (w == 16)
      return builder.getF16Type();
    if (w == 32)
      return builder.getF32Type();
    return builder.getF64Type();
  }

  mlir::Type LowerPrimTypeBool(PrimTypeBool primTypeBool) { return builder.getIntegerType(1); }

  mlir::Type LowerTypeArray(TypeArray<Shape, NoUniqueness> typeArray) {
    auto baseType = LowerPrimType(typeArray.elem);
    std::vector<int64_t> dims;
    for (auto d : typeArray.shape.dims) {
      dims.push_back(d.GetIntValue());
    }

    return mlir::RankedTensorType::get(dims, baseType);
  }

  static mlir::Type GetFloatType(mlir::Builder &builder, uint64_t width) {
    if (width == 16)
      return builder.getF16Type();
    if (width == 32)
      return builder.getF32Type();
    if (width == 64)
      return builder.getF64Type();
    Unreachable();
  }

  static FunDef GetFunction(Prog &prog, std::string fname) {
    for (auto f : prog.funs) {
      if (f.name == fname)
        return f;
    }

    Unreachable();
  }
};