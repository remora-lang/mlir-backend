#pragma once
using namespace antlr4;

void Unreachable() { throw std::runtime_error("unreachable"); }

struct FutharkTranslationVisitor {
  Prog VisitProg(FutharkParser::RootContext *ctx) {
    Prog prog;
    for (auto f : ctx->pFunDef()) {
      prog.funs.push_back(VisitFunDef(f));
    }

    return prog;
  }

  FunDef VisitFunDef(FutharkParser::PFunDefContext *ctx) {
    FunDef fun;
    fun.entry = ctx->pEntry() == nullptr ? std::nullopt : std::make_optional(VisitEntry(ctx->pEntry()));
    fun.name = VisitId(ctx->ID());
    fun.retType.push_back({RetType{VisitType(ctx->pType())}, {}});
    for (auto param : ctx->fParam())
      fun.params.push_back(VisitFParam(param));
    fun.body = VisitBody(ctx->pBody());
    return fun;
  }

  EntryPoint VisitEntry(FutharkParser::PEntryContext *ctx) {
    EntryPoint entry;
    return entry;
  }

  Type VisitType(FutharkParser::PTypeContext *ctx) {
    if (auto *pPrim = dynamic_cast<FutharkParser::TypePrimContext *>(ctx))
      return VisitTypePrim(pPrim);
    if (auto *pShape = dynamic_cast<FutharkParser::TypeShapeContext *>(ctx))
      return VisitTypeShape(pShape);
  }

  Type VisitTypePrim(FutharkParser::TypePrimContext *ctx) {
    auto primType = VisitPrimType(ctx->pPrimType());
    return Type::CreatePrim(primType);
  }

  Type VisitTypeShape(FutharkParser::TypeShapeContext *ctx) {
    auto primType = VisitPrimType(ctx->pPrimType());
    auto extShape = VisitExtShape(ctx->pExtShape());
    return Type::CreateArr(primType, extShape);
  }

  PrimType VisitPrimType(FutharkParser::PPrimTypeContext *ctx) {
    auto name = ctx->getText();
    if (name == "i8")
      return PrimType::Int8();
    if (name == "i16")
      return PrimType::Int16();
    if (name == "i32")
      return PrimType::Int32();
    if (name == "i64")
      return PrimType::Int64();
    if (name == "f16")
      return PrimType::Float16();
    if (name == "f32")
      return PrimType::Float32();
    if (name == "f64")
      return PrimType::Float64();
    Unreachable();
  }

  ShapeBase<SubExp> VisitExtShape(FutharkParser::PExtShapeContext *ctx) {
    ShapeBase<SubExp> result;
    for (auto exp : ctx->pSubExp())
      result.dims.push_back(VisitSubExp(exp));

    return result;
  }

  FParam VisitFParam(FutharkParser::FParamContext *ctx) {
    FParam fparam;
    fparam.name = VisitId(ctx->ID());
    fparam.dec.v = VisitType(ctx->pType());
    return fparam;
  }

  Body VisitBody(FutharkParser::PBodyContext *ctx) {
    if (auto *pBody = dynamic_cast<FutharkParser::BodyContext *>(ctx))
      return {VisitNormalBody(pBody)};
    if (auto *pEmptyBody = dynamic_cast<FutharkParser::EmptyBodyContext *>(ctx))
      return {VisitEmptyBody(pEmptyBody)};

    Unreachable();
  }

  Body VisitNormalBody(FutharkParser::BodyContext *ctx) {
    Body b;
    for (auto stm : ctx->pStm())
      b.stms.push_back(VisitStm(stm));
    for (auto subExp : ctx->pSubExp())
      b.result.push_back(VisitSubExp(subExp));
    return b;
  }

  Body VisitEmptyBody(FutharkParser::EmptyBodyContext *ctx) {
    Body b;
    for (auto subExp : ctx->pSubExp())
      b.result.push_back(VisitSubExp(subExp));
    return b;
  }

  Stm VisitStm(FutharkParser::PStmContext *ctx) {
    Stm stm;
    stm.pat = VisitPat(ctx->pPat());
    stm.exp = (VisitExp(ctx->pExp()));
    return stm;
  }

  Pat<LetDec> VisitPat(FutharkParser::PPatContext *ctx) {
    Pat<LetDec> pat;
    for (auto t : ctx->pPatElem()) {
      pat.elems.push_back(VisitPatElem(t));
    }
    return pat;
  }

  PatElem<LetDec> VisitPatElem(FutharkParser::PPatElemContext *ctx) {
    PatElem<LetDec> elem;
    elem.name.name = VisitId(ctx->ID());
    elem.name.tag = 0;
    elem.dec.v = VisitType(ctx->pType());
    return elem;
  }

  Exp VisitExp(FutharkParser::PExpContext *ctx) {
    if (auto *pSoac = dynamic_cast<FutharkParser::ExpSoacOpContext *>(ctx))
      return {VisitExpSoacOp(pSoac)};
    if (auto *pApply = dynamic_cast<FutharkParser::ExpApplyContext *>(ctx))
      return {VisitApply(pApply->pApply())};
    if (auto *pBasicOp = dynamic_cast<FutharkParser::ExpBasicOpContext *>(ctx))
      return {VisitExpBasicOp(pBasicOp)};
    if (auto *pSubExp = dynamic_cast<FutharkParser::ExpSubExpContext *>(ctx))
      return {VisitExpSubExp(pSubExp)};
    Unreachable();
  }

  // TODO: Parse diet and RetAls
  Exp VisitApply(FutharkParser::PApplyContext *ctx) {
    ExpApply apply;
    apply.fname = VisitId(ctx->ID());
    for (auto subExp : ctx->pSubExp())
      apply.args.push_back({VisitSubExp(subExp), Diet::Consume});
    for (auto type : ctx->pRetTypes()->pType())
      apply.retTypes.push_back({RetType{VisitType(type)}, {}});
    return {apply};
  }

  Exp VisitExpBasicOp(FutharkParser::ExpBasicOpContext *ctx) { return {ExpBasicOp{VisitBasicOp(ctx->pBasicOp())}}; }

  BasicOp VisitBasicOp(FutharkParser::PBasicOpContext *ctx) {
    if (auto *pArrayLit = dynamic_cast<FutharkParser::BasicOpArrayLitContext *>(ctx))
      return {VisitBasicOpArrayLit(pArrayLit)};
    if (auto *pBinary = dynamic_cast<FutharkParser::BasicOpBinaryContext *>(ctx)) {
      auto pb = pBinary->pBinOp();
      auto op0 = VisitSubExp(pb->pSubExp(0));
      auto op1 = VisitSubExp(pb->pSubExp(1));
      return {BasicOpBinOp{VisitBinOp(pb), op0, op1}};
    }
    if (auto *pIota = dynamic_cast<FutharkParser::BasicOpIotaContext *>(ctx))
      return {VisitBasicOpIota(pIota)};
    if (auto *pConv = dynamic_cast<FutharkParser::BasicOpConvContext *>(ctx))
      return {VisitBasicOpConvOp(pConv)};
    if (auto *pConcat = dynamic_cast<FutharkParser::BasicOpConcatContext *>(ctx))
      return {VisitBasicOpConcat(pConcat)};
    if (auto *pFlatIndex = dynamic_cast<FutharkParser::BasicOpFlatIndexContext *>(ctx))
      return {VisitBasicOpFlatIndex(pFlatIndex)};
    if (auto *pFlatIndex = dynamic_cast<FutharkParser::BasicOpReshapeContext *>(ctx))
      return {VisitBasicOpReshape(pFlatIndex)};
    if (auto *pFlatIndex = dynamic_cast<FutharkParser::BasicOpReplicateContext *>(ctx))
      return {VisitBasicOpReplicate(pFlatIndex)};
    if (auto *pFlatIndex = dynamic_cast<FutharkParser::BasicOpRearrangeContext *>(ctx))
      return {VisitBasicOpRearrange(pFlatIndex)};

    Unreachable();
  }

  BasicOpArrayLit VisitBasicOpArrayLit(FutharkParser::BasicOpArrayLitContext *ctx) {
    BasicOpArrayLit arr;
    for (auto elem : ctx->pArrayLit()->pSubExp()) {
      arr.values.push_back(VisitSubExp(elem));
    }

    arr.t = VisitType(ctx->pType());

    if (auto *val = std::get_if<TypeArray<Shape, NoUniqueness>>(&arr.t.t.v)) {
      auto subExp = ConstantSubExp{PrimValue::CreateInt(arr.values.size(), 64)};
      val->shape.dims.insert(val->shape.dims.begin(), {subExp});
    }

    return arr;
  }

  BinOp VisitBinOp(FutharkParser::PBinOpContext *ctx) {
    auto str = ctx->binaryOpcode()->getText();
    auto it = std::find_if(str.begin(), str.end(), ::isdigit);
    size_t l = std::distance(str.begin(), it);
    std::string name = str.substr(0, l);
    int width = std::stoi(str.substr(l));

    IntType intTy = IntType::Int8;
    FloatType floatTy = FloatType::Float16;
    if (width == 16) {
      intTy == IntType::Int16;
    }
    if (width == 32) {
      intTy == IntType::Int32;
      floatTy = FloatType::Float32;
    }
    if (width == 64) {
      intTy == IntType::Int64;
      floatTy = FloatType::Float64;
    }

    // TODO: Set width field
    if (name == "add")
      return {BinOpAdd{intTy}};
    if (name == "fadd")
      return {BinOpFAdd{floatTy}};
    if (name == "sub")
      return {BinOpSub{intTy}};
    if (name == "fsub")
      return {BinOpFSub{floatTy}};
    if (name == "mul")
      return {BinOpMul{intTy}};
    if (name == "smod")
      return {BinOpSMod{intTy}};
    if (name == "srem")
      return {BinOpSRem{intTy}};
    if (name == "fmul")
      return {BinOpFMul{floatTy}};
    if (name == "fdiv")
      return {BinOpFDiv{floatTy}};

    Unreachable();
  }

  BasicOpConcat VisitBasicOpConcat(FutharkParser::BasicOpConcatContext *ctx) {
    if (ctx->pSubExp().size() != 3)
      throw std::runtime_error("unrecognized concat format");

    BasicOpConcat concat;
    auto constExp = std::get<ConstantSubExp>(VisitSubExp(ctx->pSubExp(0)).v).v;
    concat.dim = std::get<Int64Value>(std::get<IntValue>(constExp.v).v).v;

    auto vname = std::get<VarSubExp>(VisitSubExp(ctx->pSubExp(1)).v).v;
    concat.arrs.push_back(vname);

    concat.total = VisitSubExp(ctx->pSubExp(2));

    return concat;
  }

  BasicOpFlatIndex VisitBasicOpFlatIndex(FutharkParser::BasicOpFlatIndexContext *ctx) {
    BasicOpFlatIndex flatIndex{};
    for (auto subexp : ctx->pSubExp()) {
      flatIndex.operands.push_back(VisitSubExp(subexp));
    }

    flatIndex.base = flatIndex.operands[0];
    flatIndex.operands.erase(flatIndex.operands.begin());
    return flatIndex;
  }

  BasicOpReshape VisitBasicOpReshape(FutharkParser::BasicOpReshapeContext *ctx) {
    BasicOpReshape reshape{};
    reshape.op0 = VisitSubExp(ctx->pSubExp());
    reshape.dimBegin = std::stoull(ctx->NUMBER(0)->getText());
    reshape.dimEnd = std::stoull(ctx->NUMBER(1)->getText());
    reshape.reshapedDim = VisitExtShape(ctx->pExtShape(0));
    reshape.remainder = VisitExtShape(ctx->pExtShape(1));
    return reshape;
  }

  BasicOpReplicate VisitBasicOpReplicate(FutharkParser::BasicOpReplicateContext *ctx) {
    BasicOpReplicate replicate;
    replicate.shape = VisitExtShape(ctx->pExtShape());
    replicate.val = VisitSubExp(ctx->pSubExp());
    return replicate;
  }

  BasicOpRearrange VisitBasicOpRearrange(FutharkParser::BasicOpRearrangeContext *ctx) {
    BasicOpRearrange rearrange;
    rearrange.arr = std::get<VarSubExp>(VisitSubExp(ctx->pSubExp()).v).v;
    for (auto number : ctx->NUMBER()) {
      rearrange.perm.push_back(std::stoull(number->getText()));
    }
    return rearrange;
  }

  BasicOp VisitBasicOpIota(FutharkParser::BasicOpIotaContext *ctx) {
    BasicOpIota iota;
    iota.n = VisitSubExp(ctx->pSubExp(0));
    iota.x = VisitSubExp(ctx->pSubExp(1));
    iota.s = VisitSubExp(ctx->pSubExp(2));

    auto text = ctx->IOTA()->getText();
    if (text == "iota16")
      iota.t = IntType::Int16;
    else if (text == "iota32")
      iota.t = IntType::Int32;
    else if (text == "iota64")
      iota.t = IntType::Int64;
    else
      Unreachable();
    return {iota};
  }

  BasicOpConvOp VisitBasicOpConvOp(FutharkParser::BasicOpConvContext *ctx) {
    ConvOp op;
    op.v = ConvOpSExt{
        std::get<PrimTypeInt>(VisitPrimType(ctx->pConvOp()->pPrimType(0)).v).t,
        std::get<PrimTypeInt>(VisitPrimType(ctx->pConvOp()->pPrimType(1)).v).t,
    };

    auto subExp = VisitSubExp(ctx->pConvOp()->pSubExp());
    return {op, subExp};
  }

  Exp VisitExpSubExp(FutharkParser::ExpSubExpContext *ctx) { return {ExpSubExp{VisitSubExp(ctx->pSubExp())}}; }

  Exp VisitExpSoacOp(FutharkParser::ExpSoacOpContext *ctx) {
    auto soac = ctx->pSoacOp();
    if (auto *pMap = dynamic_cast<FutharkParser::SoacOpMapContext *>(soac))
      return {ExpSoacOp{std::make_shared<Soac>(VisitSoacOpMap(pMap))}};
    if (auto *pRedomap = dynamic_cast<FutharkParser::SoacOpRedomapContext *>(soac))
      return {ExpSoacOp{std::make_shared<Soac>(VisitSoacOpRedomap(pRedomap))}};

    Unreachable();
  }

  Soac VisitSoacOpMap(FutharkParser::SoacOpMapContext *ctx) {
    SoacScrema screma;
    screma.w = VisitSubExp(ctx->pScrema()->pSubExp());
    for (auto id : ctx->pScrema()->ID()) {
      screma.arrs.push_back(VName{id->getText()});
    }

    screma.form = VisitMapForm(ctx->pMapForm());
    return {screma};
  }

  ScremaForm VisitMapForm(FutharkParser::PMapFormContext *ctx) {
    ScremaForm form;
    form.scremaLambda = VisitLambda(ctx->pLambda());
    return form;
  }

  Soac VisitSoacOpRedomap(FutharkParser::SoacOpRedomapContext *ctx) {
    SoacScrema screma;
    screma.w = VisitSubExp(ctx->pScrema()->pSubExp());
    for (auto id : ctx->pScrema()->ID()) {
      screma.arrs.push_back(VName{id->getText()});
    }

    screma.form = VisitRedomapForm(ctx->pRedomapForm());
    return {screma};
  }

  ScremaForm VisitRedomapForm(FutharkParser::PRedomapFormContext *ctx) {
    ScremaForm form;
    form.scremaLambda = VisitLambda(ctx->pLambda());
    for (auto reduce : ctx->pReduce())
      form.scremaReduces.push_back(VisitReduce(reduce));
    return form;
  }

  Lambda VisitLambda(FutharkParser::PLambdaContext *ctx) {
    Lambda lambda;
    for (auto param : ctx->pLParam()) {
      LParam l;
      l.name = VisitId(param->ID());
      l.dec = {VisitType(param->pType())};
      lambda.params.push_back(l);
    }

    for (auto t : ctx->pTypes()->pType())
      lambda.ret.push_back(VisitType(t));
    lambda.body = VisitBody(ctx->pBody());
    return lambda;
  }

  Reduce VisitReduce(FutharkParser::PReduceContext *ctx) {
    Reduce reduce;
    // TODO: Parse commutativity
    reduce.comm = Commutativity::Commutative;
    reduce.lambda = VisitLambda(ctx->pLambda());
    for (auto subExp : ctx->pSubExp())
      reduce.neutral.push_back(VisitSubExp(subExp));

    return reduce;
  }

  SubExp VisitSubExp(FutharkParser::PSubExpContext *ctx) {
    if (auto *pConst = dynamic_cast<FutharkParser::ConstantSubExpressionContext *>(ctx))
      return {ConstantSubExp{VisitPrimValue(pConst->pPrimValue())}};
    if (auto *pVar = dynamic_cast<FutharkParser::VarSubExpContext *>(ctx))
      return {VarSubExp{VisitId(pVar->ID())}};

    Unreachable();
  }

  PrimValue VisitPrimValue(FutharkParser::PPrimValueContext *ctx) {
    if (auto *pInt = dynamic_cast<FutharkParser::PrimValueIntegerContext *>(ctx))
      return {VisitPrimValueInteger(pInt)};
    if (auto *pFloat = dynamic_cast<FutharkParser::PrimValueFloatContext *>(ctx))
      return {VisitPrimValueFloat(pFloat)};

    Unreachable();
  }

  IntValue VisitPrimValueInteger(FutharkParser::PrimValueIntegerContext *ctx) {
    auto v = std::stoull(ctx->NUMBER()->getText());
    auto w = ctx->INTEGER_TYPE()->getText();
    if (w == "i8")
      return IntValue{Int8Value{(int8_t)v}};
    if (w == "i16")
      return IntValue{Int16Value{(int16_t)v}};
    if (w == "i32")
      return IntValue{Int32Value{(int32_t)v}};
    if (w == "i64")
      return IntValue{Int64Value{(int64_t)v}};

    Unreachable();
  }

  FloatValue VisitPrimValueFloat(FutharkParser::PrimValueFloatContext *ctx) {
    auto v = std::stoull(ctx->FLOAT()->getText());
    auto w = ctx->FLOAT_TYPE()->getText();
    if (w == "f16")
      return FloatValue{Float16Value{(uint16_t)v}};
    if (w == "f32")
      return FloatValue{Float32Value{(float)v}};
    if (w == "f64")
      return FloatValue{Float64Value{(double)v}};

    Unreachable();
  }

  std::string VisitId(antlr4::tree::TerminalNode *ctx) { return ctx->getText(); }
};

Prog ParseFuthark(std::ifstream &file) {
  ANTLRInputStream input(file);
  FutharkLexer lexer(&input);
  CommonTokenStream tokens(&lexer);
  tokens.fill();
  FutharkParser parser(&tokens);
  auto tree = parser.root();

  FutharkTranslationVisitor visitor;
  return visitor.VisitProg(tree);
}