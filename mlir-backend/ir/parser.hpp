#pragma once
#include "FutharkLexer.h"
#include "FutharkParser.h"
#include "blackbox.hpp"
#include "segop.hpp"
#include "syntax.hpp"
#include "utils.hpp"
#include <limits>

using namespace antlr4;

struct FutharkTranslationVisitor {
  Prog VisitProg(FutharkParser::RootContext *ctx) {
    Prog prog;
    for (auto stm : ctx->pStm())
      prog.consts.push_back(VisitStm(stm));
    for (auto f : ctx->pFunDef()) {
      prog.funs.push_back(VisitFunDef(f));
    }

    return prog;
  }

  FunDef VisitFunDef(FutharkParser::PFunDefContext *ctx) {
    FunDef fun;
    fun.entry = ctx->pEntry() == nullptr
                    ? std::nullopt
                    : std::make_optional(VisitEntry(ctx->pEntry()));
    fun.name = VisitId(ctx->ID());
    for (auto type : ctx->pType())
      fun.retType.push_back({RetType{VisitType(type)}, {}});
    for (auto param : ctx->fParam())
      fun.params.push_back(VisitFParam(param));

    for (auto *attr : ctx->pAttr())
      fun.attrs.attrs.push_back(VisitAttr(attr));

    if (!IsBlackBox(fun.attrs))
      fun.body = VisitBody(ctx->pBody());
    return fun;
  }

  Attr VisitAttr(FutharkParser::PAttrContext *ctx) {
    auto ids = ctx->ID();
    std::string name = ids[0]->getText();
    if (ids.size() == 1)
      return Attr{AtomAttr{name}};
    CompAttr comp{name, {}};
    for (size_t i = 1; i < ids.size(); ++i)
      comp.args.push_back(Attr{AtomAttr{ids[i]->getText()}});
    return Attr{comp};
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
    if (auto *pUnitArr =
            dynamic_cast<FutharkParser::TypeUnitArrayContext *>(ctx))
      return Type::CreateArr(PrimType{PrimTypeUnit{}},
                             VisitExtShape(pUnitArr->pExtShape()));
    if (dynamic_cast<FutharkParser::TypeUnitContext *>(ctx))
      return Type::CreatePrim(PrimType{PrimTypeUnit{}});
    if (auto *pAcc = dynamic_cast<FutharkParser::TypeAccContext *>(ctx))
      return VisitTypeAcc(pAcc);
    Undefined();
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

  Type VisitTypeAcc(FutharkParser::TypeAccContext *ctx) {
    VName cert{VisitId(ctx->ID()), 0};
    Shape ispace = VisitExtShape(ctx->pExtShape());
    std::vector<Type> elementTypes;
    for (auto type : ctx->pTypes()->pType())
      elementTypes.push_back(VisitType(type));
    return Type::CreateAcc(cert, ispace, elementTypes);
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
    if (name == "bool")
      return {PrimTypeBool{}};
    Undefined();
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

    Undefined();
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
    if (auto *pSegOp = dynamic_cast<FutharkParser::ExpSegOpContext *>(ctx))
      return {VisitExpSegOp(pSegOp)};
    if (auto *pApply = dynamic_cast<FutharkParser::ExpApplyContext *>(ctx))
      return {VisitApply(pApply->pApply())};
    if (auto *pBasicOp = dynamic_cast<FutharkParser::ExpBasicOpContext *>(ctx))
      return {VisitExpBasicOp(pBasicOp)};
    if (auto *pSizeOp = dynamic_cast<FutharkParser::ExpSizeOpContext *>(ctx))
      return {VisitSizeOp(pSizeOp->pSizeOp())};
    if (auto *pIf = dynamic_cast<FutharkParser::ExpIfContext *>(ctx))
      return VisitExpIf(pIf);
    if (auto *pLoop = dynamic_cast<FutharkParser::ExpLoopContext *>(ctx))
      return VisitExpLoop(pLoop->pLoop());
    if (auto *pGpu = dynamic_cast<FutharkParser::ExpGpuBodyContext *>(ctx))
      return VisitExpGpuBody(pGpu);
    if (auto *pWithAcc =
            dynamic_cast<FutharkParser::ExpWithAccContext *>(ctx))
      return VisitExpWithAcc(pWithAcc);
    if (auto *pSubExp = dynamic_cast<FutharkParser::ExpSubExpContext *>(ctx))
      return {VisitExpSubExp(pSubExp)};
    Undefined();
  }

  Exp VisitExpIf(FutharkParser::ExpIfContext *ctx) {
    ExpIf e;
    if (auto *sort = ctx->MATCH_SORT())
      e.sort =
          sort->getText() == "<equiv>" ? MatchSort::Equiv : MatchSort::Fallback;
    e.cond = VisitSubExp(ctx->pSubExp());
    e.then_body = std::make_shared<Body>(VisitCaseBody(ctx->pCaseBody(0)));
    e.else_body = std::make_shared<Body>(VisitCaseBody(ctx->pCaseBody(1)));
    for (auto t : ctx->pTypes()->pType())
      e.retType.push_back(VisitType(t));
    return {e};
  }

  Exp VisitExpLoop(FutharkParser::PLoopContext *ctx) {
    ExpLoop loop;
    // The loop form has its own rule, so the bound of a `for` is not among
    // these subexpressions: they are exactly the initial values.
    auto params = ctx->pLoopParam();
    auto inits = ctx->pSubExp();
    if (params.size() != inits.size())
      throw std::runtime_error("loop has a different number of parameters "
                               "than initial values");
    for (size_t i = 0; i < params.size(); i++) {
      FParam param;
      param.name = VisitId(params[i]->ID());
      param.dec.v = VisitType(params[i]->pType());
      loop.merge.push_back({param, VisitSubExp(inits[i])});
    }
    loop.form = VisitLoopForm(ctx->pLoopForm());
    loop.body = std::make_shared<Body>(VisitCaseBody(ctx->pCaseBody()));
    return {loop};
  }

  LoopForm VisitLoopForm(FutharkParser::PLoopFormContext *ctx) {
    if (auto *pFor = dynamic_cast<FutharkParser::LoopForContext *>(ctx)) {
      ForLoop forLoop;
      forLoop.i = VName{VisitId(pFor->ID())};
      forLoop.t = std::get<PrimTypeInt>(VisitPrimType(pFor->pPrimType()).v).t;
      forLoop.bound = VisitSubExp(pFor->pSubExp());
      return {forLoop};
    }
    if (auto *pWhile = dynamic_cast<FutharkParser::LoopWhileContext *>(ctx))
      return {WhileLoop{VName{VisitId(pWhile->ID())}}};

    Undefined();
  }

  Exp VisitExpGpuBody(FutharkParser::ExpGpuBodyContext *ctx) {
    GPUBody g;
    for (auto t : ctx->pTypes()->pType())
      g.retType.push_back(VisitType(t));
    g.body = std::make_shared<Body>(VisitCaseBody(ctx->pCaseBody()));
    return {ExpHostOp{HostOp{g}}};
  }

  Body VisitCaseBody(FutharkParser::PCaseBodyContext *ctx) {
    Body b;
    if (auto *full = dynamic_cast<FutharkParser::CaseBodyFullContext *>(ctx)) {
      for (auto stm : full->pStm())
        b.stms.push_back(VisitStm(stm));
      for (auto se : full->pSubExp())
        b.result.push_back(VisitSubExp(se));
      return b;
    }
    if (auto *res = dynamic_cast<FutharkParser::CaseBodyResultContext *>(ctx)) {
      for (auto se : res->pSubExp())
        b.result.push_back(VisitSubExp(se));
      return b;
    }
    Undefined();
  }

  Exp VisitSizeOp(FutharkParser::PSizeOpContext *ctx) {
    return {ExpHostOp{HostOp{VisitSizeOpInner(ctx)}}};
  }

  SizeOp VisitSizeOpInner(FutharkParser::PSizeOpContext *ctx) {
    if (auto *pGet = dynamic_cast<FutharkParser::SizeOpGetSizeContext *>(ctx))
      return {GetSize{VisitId(pGet->ID()), VisitSizeClass(pGet->pSizeClass())}};
    if (auto *pMax =
            dynamic_cast<FutharkParser::SizeOpGetSizeMaxContext *>(ctx))
      return {GetSizeMax{VisitSizeClass(pMax->pSizeClass())}};
    if (auto *pCmp = dynamic_cast<FutharkParser::SizeOpCmpSizeLeContext *>(ctx))
      return {CmpSizeLe{VisitId(pCmp->ID()),
                        VisitSizeClass(pCmp->pSizeClass()),
                        VisitSubExp(pCmp->pSubExp())}};

    Undefined();
  }

  SizeClass VisitSizeClass(FutharkParser::PSizeClassContext *ctx) {
    if (auto *pThreshold =
            dynamic_cast<FutharkParser::SizeClassThresholdContext *>(ctx)) {
      SizeThreshold threshold;
      if (auto *number = pThreshold->NUMBER())
        threshold.def = std::stoll(number->getText());
      return {threshold};
    }
    if (auto *pNamed =
            dynamic_cast<FutharkParser::SizeClassNamedContext *>(ctx))
      return SizeClassFromName(VisitId(pNamed->ID()));

    Undefined();
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

  Exp VisitExpBasicOp(FutharkParser::ExpBasicOpContext *ctx) {
    return {ExpBasicOp{VisitBasicOp(ctx->pBasicOp())}};
  }

  BasicOpUpdateAcc VisitBasicOpUpdateAcc(
    FutharkParser::BasicOpUpdateAccContext *ctx) {
    BasicOpUpdateAcc update;
    update.safety = Safe;
    update.acc = VName{VisitId(ctx->ID()), 0};
    update.indices.push_back(VisitSubExp(ctx->pSubExp(0)));
    update.values.push_back(VisitSubExp(ctx->pSubExp(1)));
    return update;
  }

  BasicOp VisitBasicOp(FutharkParser::PBasicOpContext *ctx) {
    if (auto *pArrayLit =
            dynamic_cast<FutharkParser::BasicOpArrayLitContext *>(ctx))
      return {VisitBasicOpArrayLit(pArrayLit)};
    if (auto *pBinary =
            dynamic_cast<FutharkParser::BasicOpBinaryContext *>(ctx)) {
      auto pb = pBinary->pBinOp();
      auto op0 = VisitSubExp(pb->pSubExp(0));
      auto op1 = VisitSubExp(pb->pSubExp(1));
      return {BasicOpBinOp{VisitBinOp(pb), op0, op1}};
    }
    if (auto *pIota = dynamic_cast<FutharkParser::BasicOpIotaContext *>(ctx))
      return {VisitBasicOpIota(pIota)};
    if (auto *pConv = dynamic_cast<FutharkParser::BasicOpConvContext *>(ctx))
      return {VisitBasicOpConvOp(pConv)};
    if (auto *pConcat =
            dynamic_cast<FutharkParser::BasicOpConcatContext *>(ctx))
      return {VisitBasicOpConcat(pConcat)};
    if (auto *pIndex = dynamic_cast<FutharkParser::BasicOpIndexContext *>(ctx))
      return {VisitBasicOpIndex(pIndex)};
    if (auto *pManifest =
            dynamic_cast<FutharkParser::BasicOpManifestContext *>(ctx))
      return {VisitBasicOpManifest(pManifest)};
    if (auto *pFlatIndex =
            dynamic_cast<FutharkParser::BasicOpReshapeContext *>(ctx))
      return {VisitBasicOpReshape(pFlatIndex)};
    if (auto *pFlatIndex =
            dynamic_cast<FutharkParser::BasicOpReplicateContext *>(ctx))
      return {VisitBasicOpReplicate(pFlatIndex)};
    if (auto *pFlatIndex =
            dynamic_cast<FutharkParser::BasicOpRearrangeContext *>(ctx))
      return {VisitBasicOpRearrange(pFlatIndex)};
    if (auto *pScratch =
            dynamic_cast<FutharkParser::BasicOpScratchContext *>(ctx))
      return {VisitBasicOpScratch(pScratch)};
    if (auto *pCmp = dynamic_cast<FutharkParser::BasicOpCmpContext *>(ctx)) {
      auto pc = pCmp->pCmpOp();
      auto op0 = VisitSubExp(pc->pSubExp(0));
      auto op1 = VisitSubExp(pc->pSubExp(1));
      return {BasicOpCmpOp{VisitCmpOp(pc), op0, op1}};
    }
    if (dynamic_cast<FutharkParser::BasicOpAssertContext *>(ctx))
      return {BasicOpAssert{}};
    if (auto *pUpdateAcc =
            dynamic_cast<FutharkParser::BasicOpUpdateAccContext *>(ctx))
      return {VisitBasicOpUpdateAcc(pUpdateAcc)};

    Undefined();
  }

  CmpOp VisitCmpOp(FutharkParser::PCmpOpContext *ctx) {
    auto str = ctx->cmpOpcode()->getText();
    auto it = std::find_if(str.begin(), str.end(), ::isdigit);
    int width = it == str.end()
                    ? 0
                    : std::stoi(str.substr(std::distance(str.begin(), it)));
    IntType intTy = width == 8    ? IntType::Int8
                    : width == 16 ? IntType::Int16
                    : width == 32 ? IntType::Int32
                                  : IntType::Int64;
    FloatType floatTy = width == 16   ? FloatType::Float16
                        : width == 32 ? FloatType::Float32
                                      : FloatType::Float64;

    if (str.rfind("eq_", 0) == 0) {
      if (str.find("bool") != std::string::npos)
        return {CmpOpEq{PrimType{PrimTypeBool{}}}};
      if (str.find('f') != std::string::npos)
        return {CmpOpEq{PrimType::Float(width)}};
      return {CmpOpEq{PrimType::Int(width)}};
    }
    if (str.rfind("slt", 0) == 0)
      return {CmpOpSlt{intTy}};
    if (str.rfind("sle", 0) == 0)
      return {CmpOpSle{intTy}};
    if (str.rfind("ult", 0) == 0)
      return {CmpOpUlt{intTy}};
    if (str.rfind("ule", 0) == 0)
      return {CmpOpUle{intTy}};
    // Ordered float comparisons: lt32, le64, ...
    if (str.rfind("lt", 0) == 0)
      return {CmpOpFlt{floatTy}};
    if (str.rfind("le", 0) == 0)
      return {CmpOpFle{floatTy}};

    Undefined();
  }

  BasicOpArrayLit
  VisitBasicOpArrayLit(FutharkParser::BasicOpArrayLitContext *ctx) {
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

    // The logical operators are the only ones without a width suffix.
    if (name == "logand")
      return {BinOpLogAnd{}};
    if (name == "logor")
      return {BinOpLogOr{}};

    int width = std::stoi(str.substr(l));

    IntType intTy = IntType::Int8;
    FloatType floatTy = FloatType::Float16;
    if (width == 16) {
      intTy = IntType::Int16;
    }
    if (width == 32) {
      intTy = IntType::Int32;
      floatTy = FloatType::Float32;
    }
    if (width == 64) {
      intTy = IntType::Int64;
      floatTy = FloatType::Float64;
    }

    // `_nw` is Futhark's undefined-on-overflow arithmetic; it lowers the same
    // way as the wrapping kind.
    if (name == "add")
      return {BinOpAdd{intTy, OverflowWrap}};
    if (name == "add_nw")
      return {BinOpAdd{intTy, OverflowUndef}};
    if (name == "sub")
      return {BinOpSub{intTy, OverflowWrap}};
    if (name == "sub_nw")
      return {BinOpSub{intTy, OverflowUndef}};
    if (name == "mul")
      return {BinOpMul{intTy, OverflowWrap}};
    if (name == "mul_nw")
      return {BinOpMul{intTy, OverflowUndef}};
    if (name == "udiv")
      return {BinOpUDiv{intTy}};
    if (name == "udiv_up")
      return {BinOpUDivUp{intTy}};
    if (name == "sdiv")
      return {BinOpSDiv{intTy}};
    if (name == "sdiv_up")
      return {BinOpSDivUp{intTy}};
    if (name == "umod")
      return {BinOpUMod{intTy}};
    if (name == "smod")
      return {BinOpSMod{intTy}};
    if (name == "squot")
      return {BinOpSQuot{intTy}};
    if (name == "srem")
      return {BinOpSRem{intTy}};
    if (name == "smin")
      return {BinOpSMin{intTy}};
    if (name == "umin")
      return {BinOpUMin{intTy}};
    if (name == "smax")
      return {BinOpSMax{intTy}};
    if (name == "umax")
      return {BinOpUMax{intTy}};
    if (name == "shl")
      return {BinOpShl{intTy}};
    if (name == "lshr")
      return {BinOpLShr{intTy}};
    if (name == "ashr")
      return {BinOpAShr{intTy}};
    if (name == "and")
      return {BinOpAnd{intTy}};
    if (name == "or")
      return {BinOpOr{intTy}};
    if (name == "xor")
      return {BinOpXor{intTy}};
    if (name == "pow")
      return {BinOpPow{intTy}};

    if (name == "fadd")
      return {BinOpFAdd{floatTy}};
    if (name == "fsub")
      return {BinOpFSub{floatTy}};
    if (name == "fmul")
      return {BinOpFMul{floatTy}};
    if (name == "fdiv")
      return {BinOpFDiv{floatTy}};
    if (name == "fmod")
      return {BinOpFMod{floatTy}};
    if (name == "fmin")
      return {BinOpFMin{floatTy}};
    if (name == "fmax")
      return {BinOpFMax{floatTy}};
    if (name == "fpow")
      return {BinOpFPow{floatTy}};

    Undefined();
  }

  BasicOpConcat VisitBasicOpConcat(FutharkParser::BasicOpConcatContext *ctx) {
    // `concat@d(w, x, y, ...)`: the dimension is part of the opcode, the first
    // operand is that dimension's size in the result, and the rest are the
    // arrays being concatenated.
    auto subExps = ctx->pSubExp();
    if (subExps.size() < 2)
      throw std::runtime_error("unrecognized concat format");

    BasicOpConcat concat;
    auto opcode = ctx->CONCAT()->getText();
    concat.dim = std::stoi(opcode.substr(opcode.find('@') + 1));
    concat.total = VisitSubExp(subExps[0]);
    for (size_t i = 1; i < subExps.size(); i++)
      concat.arrs.push_back(VisitSubExp(subExps[i]).GetVName());

    return concat;
  }

  BasicOpIndex VisitBasicOpIndex(FutharkParser::BasicOpIndexContext *ctx) {
    BasicOpIndex index;
    index.vName = VisitSubExp(ctx->pSubExp()).GetVName();
    for (auto dim : ctx->pDimIndex())
      index.slice.dims.push_back(VisitDimIndex(dim));
    return index;
  }

  DimIndex<SubExp> VisitDimIndex(FutharkParser::PDimIndexContext *ctx) {
    if (auto *pSlice = dynamic_cast<FutharkParser::DimSliceContext *>(ctx))
      return {DimSlice<SubExp>{VisitSubExp(pSlice->pSubExp(0)),
                               VisitSubExp(pSlice->pSubExp(1)),
                               VisitSubExp(pSlice->pSubExp(2))}};
    if (auto *pFix = dynamic_cast<FutharkParser::DimFixContext *>(ctx))
      return {DimFix<SubExp>{VisitSubExp(pFix->pSubExp())}};

    Undefined();
  }

  BasicOpManifest
  VisitBasicOpManifest(FutharkParser::BasicOpManifestContext *ctx) {
    BasicOpManifest manifest;
    manifest.arr = VisitSubExp(ctx->pSubExp()).GetVName();
    for (auto number : ctx->NUMBER())
      manifest.perm.push_back(std::stoi(number->getText()));
    return manifest;
  }

  BasicOpReshape
  VisitBasicOpReshape(FutharkParser::BasicOpReshapeContext *ctx) {
    BasicOpReshape reshape{};
    reshape.op0 = VisitSubExp(ctx->pSubExp());
    reshape.dimBegin = std::stoull(ctx->NUMBER(0)->getText());
    reshape.dimEnd = std::stoull(ctx->NUMBER(1)->getText());
    reshape.reshapedDim = VisitExtShape(ctx->pExtShape(0));
    reshape.remainder = VisitExtShape(ctx->pExtShape(1));
    return reshape;
  }

  BasicOpReplicate
  VisitBasicOpReplicate(FutharkParser::BasicOpReplicateContext *ctx) {
    BasicOpReplicate replicate;
    replicate.shape = VisitExtShape(ctx->pExtShape());
    replicate.val = VisitSubExp(ctx->pSubExp());
    return replicate;
  }

  BasicOpRearrange
  VisitBasicOpRearrange(FutharkParser::BasicOpRearrangeContext *ctx) {
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
      Undefined();
    return {iota};
  }

  BasicOpScratch
  VisitBasicOpScratch(FutharkParser::BasicOpScratchContext *ctx) {
    BasicOpScratch scratch;
    scratch.type = VisitPrimType(ctx->pPrimType());
    for (auto subExp : ctx->pSubExp())
      scratch.dims.push_back(VisitSubExp(subExp));
    return scratch;
  }

  BasicOpConvOp VisitBasicOpConvOp(FutharkParser::BasicOpConvContext *ctx) {
    // All conversions print as `<opcode> <from> <x> to <to>`.
    auto *conv = ctx->pConvOp();
    auto opcode = conv->convOpcode()->getText();
    auto from = VisitPrimType(conv->pPrimType(0));
    auto to = VisitPrimType(conv->pPrimType(1));

    auto asInt = [](const PrimType &t) { return std::get<PrimTypeInt>(t.v).t; };
    auto asFloat = [](const PrimType &t) {
      return std::get<PrimTypeFloat>(t.v).t;
    };

    ConvOp op;
    if (opcode == "zext")
      op.v = ConvOpZExt{asInt(from), asInt(to)};
    else if (opcode == "sext")
      op.v = ConvOpSExt{asInt(from), asInt(to)};
    else if (opcode == "fpconv")
      op.v = ConvOpFPConv{asFloat(from), asFloat(to)};
    else if (opcode == "fptoui")
      op.v = ConvOpFPToUI{asFloat(from), asInt(to)};
    else if (opcode == "fptosi")
      op.v = ConvOpFPToSI{asFloat(from), asInt(to)};
    else if (opcode == "uitofp")
      op.v = ConvOpUIToFP{asInt(from), asFloat(to)};
    else if (opcode == "sitofp")
      op.v = ConvOpSIToFP{asInt(from), asFloat(to)};
    else if (opcode == "itob")
      op.v = ConvOpIToB{asInt(from)};
    else if (opcode == "btoi")
      op.v = ConvOpBToI{asInt(to)};
    else if (opcode == "ftob")
      op.v = ConvOpFToB{asFloat(from)};
    else if (opcode == "btof")
      op.v = ConvOpBToF{asFloat(to)};
    else
      Undefined();

    return {op, VisitSubExp(conv->pSubExp())};
  }

  Exp VisitExpSubExp(FutharkParser::ExpSubExpContext *ctx) {
    return {ExpSubExp{VisitSubExp(ctx->pSubExp())}};
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

  WithAccInput
  VisitWithAccInput(FutharkParser::PWithAccInputContext *ctx) {
    WithAccInput input;
    input.ispace = VisitExtShape(ctx->pExtShape());
    for (auto array : ctx->ID())
      input.arrays.push_back(VName{VisitId(array), 0});
    if (auto op = ctx->pWithAccOp()) {
      std::vector<SubExp> neutral;
      for (auto subExp : op->pSubExp())
        neutral.push_back(VisitSubExp(subExp));
      input.op = std::make_pair(
          std::make_shared<Lambda>(VisitLambda(op->pLambda())), neutral);
    }
    return input;
  }

  Exp VisitExpWithAcc(FutharkParser::ExpWithAccContext *ctx) {
    ExpWithAcc withAcc;
    for (auto input : ctx->pWithAcc()->pWithAccInput())
      withAcc.inputs.push_back(VisitWithAccInput(input));
    withAcc.lambda =
        std::make_shared<Lambda>(VisitLambda(ctx->pWithAcc()->pLambda()));
    return {withAcc};
  }

  Exp VisitExpSegOp(FutharkParser::ExpSegOpContext *ctx) {
    auto segop = ctx->pSegOp();
    if (auto *pSegMap = dynamic_cast<FutharkParser::SegMapContext *>(segop))
      return {ExpHostOp{HostOp{std::make_shared<SegOp>(VisitSegMap(pSegMap))}}};
    if (auto *pSegRed = dynamic_cast<FutharkParser::SegRedContext *>(segop))
      return {ExpHostOp{HostOp{std::make_shared<SegOp>(VisitSegRed(pSegRed))}}};
    if (auto *pSegScan = dynamic_cast<FutharkParser::SegScanContext *>(segop))
      return {
          ExpHostOp{HostOp{std::make_shared<SegOp>(VisitSegScan(pSegScan))}}};
    if (auto *pSegHist = dynamic_cast<FutharkParser::SegHistContext *>(segop))
      return {
          ExpHostOp{HostOp{std::make_shared<SegOp>(VisitSegHist(pSegHist))}}};
    Undefined();
  }

  SegLevel VisitSegLevel(FutharkParser::PSegLevelContext *ctx) {
    if (auto *pThread =
            dynamic_cast<FutharkParser::SegLevelThreadContext *>(ctx))
      return {SegThread{VisitSegVirt(pThread->pSegVirt())}};
    if (auto *pBlock = dynamic_cast<FutharkParser::SegLevelBlockContext *>(ctx))
      return {SegBlock{VisitSegVirt(pBlock->pSegVirt())}};
    if (dynamic_cast<FutharkParser::SegLevelThreadInBlockContext *>(ctx))
      return {SegThreadInBlock{}};

    Undefined();
  }

  SegVirt VisitSegVirt(FutharkParser::PSegVirtContext *ctx) {
    return ctx->getText() == "virtualise" ? SegVirt::Virt : SegVirt::NoVirt;
  }

  SegOp VisitSegMap(FutharkParser::SegMapContext *ctx) {
    SegMap segmap;

    segmap.lvl = VisitSegLevel(ctx->pSegLevel());
    segmap.space = VisitSegSpace(ctx->pSegSpace());
    for (auto t : ctx->pTypes()->pType()) {
      segmap.ret.push_back(VisitType(t));
    }
    segmap.body = VisitKernelBody(ctx->pKernelBody());

    return {segmap};
  }

  SegOp VisitSegRed(FutharkParser::SegRedContext *ctx) {
    SegRed segred;
    segred.lvl = VisitSegLevel(ctx->pSegLevel());
    segred.space = VisitSegSpace(ctx->pSegSpace());
    for (auto t : ctx->pTypes()->pType())
      segred.ret.push_back(VisitType(t));
    segred.body = VisitKernelBody(ctx->pKernelBody());
    for (auto op : ctx->pSegBinOp())
      segred.ops.push_back(VisitSegBinOp(op));
    return {segred};
  }

  SegOp VisitSegScan(FutharkParser::SegScanContext *ctx) {
    SegScan segscan;
    segscan.lvl = VisitSegLevel(ctx->pSegLevel());
    segscan.space = VisitSegSpace(ctx->pSegSpace());
    for (auto t : ctx->pTypes()->pType())
      segscan.ret.push_back(VisitType(t));
    segscan.body = VisitKernelBody(ctx->pKernelBody());
    for (auto op : ctx->pSegBinOp())
      segscan.ops.push_back(VisitSegBinOp(op));
    segscan.post_op = VisitSegPostOp(ctx->pSegPostOp());
    return {segscan};
  }

  SegOp VisitSegHist(FutharkParser::SegHistContext *ctx) {
    SegHist seghist;
    seghist.lvl = VisitSegLevel(ctx->pSegLevel());
    seghist.space = VisitSegSpace(ctx->pSegSpace());
    for (auto t : ctx->pTypes()->pType())
      seghist.ret.push_back(VisitType(t));
    seghist.body = VisitKernelBody(ctx->pKernelBody());
    for (auto op : ctx->pHistOp())
      seghist.ops.push_back(VisitHistOp(op));
    return {seghist};
  }

  HistOp VisitHistOp(FutharkParser::PHistOpContext *ctx) {
    HistOp op;
    // pExtShape[0] is the histogram shape; the optional pExtShape[1] is the
    // operator shape (empty for a scalar operator).
    op.shape = VisitExtShape(ctx->pExtShape(0));
    op.raceFactor = VisitSubExp(ctx->pSubExp());
    for (auto d : ctx->pHistDest()->pSubExp())
      op.dest.push_back(std::get<VarSubExp>(VisitSubExp(d).v).v);
    for (auto ne : ctx->pHistNeutral()->pSubExp())
      op.neutral.push_back(VisitSubExp(ne));
    if (ctx->pExtShape().size() > 1)
      op.opShape = VisitExtShape(ctx->pExtShape(1)).dims;
    op.lambda = VisitLambda(ctx->pLambda());
    return op;
  }

  SegPostOp VisitSegPostOp(FutharkParser::PSegPostOpContext *ctx) {
    return {VisitLambda(ctx->pLambda())};
  }

  SegBinOp VisitSegBinOp(FutharkParser::PSegBinOpContext *ctx) {
    SegBinOp op;
    op.comm = ctx->pComm()->getText() == "commutative"
                  ? Commutativity::Commutative
                  : Commutativity::Noncommutative;
    op.lambda = VisitLambda(ctx->pLambda());
    for (auto subExp : ctx->pSubExp())
      op.neutral.push_back(VisitSubExp(subExp));
    if (ctx->pExtShape())
      op.shape = VisitExtShape(ctx->pExtShape());
    return op;
  }

  KernelBody VisitKernelBody(FutharkParser::PKernelBodyContext *ctx) {
    if (auto *pKernelBody =
            dynamic_cast<FutharkParser::KernelBodyContext *>(ctx))
      return {VisitNormalKernelBody(pKernelBody)};
    if (auto *pEmptyKernelBody =
            dynamic_cast<FutharkParser::EmptyKernelBodyContext *>(ctx))
      return {VisitEmptyKernelBody(pEmptyKernelBody)};

    Undefined();
  }

  KernelBody VisitNormalKernelBody(FutharkParser::KernelBodyContext *ctx) {
    KernelBody b;
    for (auto stm : ctx->pStm())
      b.stms.push_back(VisitStm(stm));
    for (auto subExp : ctx->pSubExp())
      b.result.push_back(KernelResult{VisitSubExp(subExp)});
    return b;
  }

  KernelBody VisitEmptyKernelBody(FutharkParser::EmptyKernelBodyContext *ctx) {
    KernelBody b;
    for (auto subExp : ctx->pSubExp())
      b.result.push_back(KernelResult{VisitSubExp(subExp)});
    return b;
  }

  SegSpace VisitSegSpace(FutharkParser::PSegSpaceContext *ctx) {
    SegSpace space;
    size_t n = ctx->pSubExp().size();
    for (size_t i = 0; i < n; i++) {
      std::string name = ctx->ID()[i]->getText();
      SubExp size = VisitSubExp(ctx->pSubExp()[i]);
      space.dims.push_back(std::make_tuple(name, size));
    }
    space.flat_id = ctx->ID()[n]->getText();
    return space;
  }

  SubExp VisitSubExp(FutharkParser::PSubExpContext *ctx) {
    if (auto *pConst =
            dynamic_cast<FutharkParser::ConstantSubExpressionContext *>(ctx))
      return {ConstantSubExp{VisitPrimValue(pConst->pPrimValue())}};
    if (auto *pVar = dynamic_cast<FutharkParser::VarSubExpContext *>(ctx))
      return {VarSubExp{VisitId(pVar->ID())}};

    Undefined();
  }

  PrimValue VisitPrimValue(FutharkParser::PPrimValueContext *ctx) {
    if (auto *pInt =
            dynamic_cast<FutharkParser::PrimValueIntegerContext *>(ctx))
      return {VisitPrimValueInteger(pInt)};
    if (auto *pFloat =
            dynamic_cast<FutharkParser::PrimValueFloatContext *>(ctx))
      return {VisitPrimValueFloat(pFloat)};
    if (auto *pSpecial =
            dynamic_cast<FutharkParser::PrimValueFloatSpecialContext *>(ctx))
      return {VisitPrimValueFloatSpecial(pSpecial)};
    if (auto *pBool =
            dynamic_cast<FutharkParser::PrimValueBoolContext *>(ctx))
      return {BoolValue{pBool->getText() == "true"}};

    Undefined();
  }

  IntValue VisitPrimValueInteger(FutharkParser::PrimValueIntegerContext *ctx) {
    // NUMBER carries its own optional sign (e.g. "-1"), matching Futhark's
    // `show`-based printing.
    int64_t v = std::stoll(ctx->NUMBER()->getText());
    auto w = ctx->INTEGER_TYPE()->getText();
    if (w == "i8")
      return IntValue{Int8Value{(int8_t)v}};
    if (w == "i16")
      return IntValue{Int16Value{(int16_t)v}};
    if (w == "i32")
      return IntValue{Int32Value{(int32_t)v}};
    if (w == "i64")
      return IntValue{Int64Value{(int64_t)v}};

    Undefined();
  }

  FloatValue VisitPrimValueFloat(FutharkParser::PrimValueFloatContext *ctx) {
    // FLOAT carries its own optional sign (e.g. "-3.0").
    double v = std::stod(ctx->FLOAT()->getText());
    auto w = ctx->FLOAT_TYPE()->getText();
    if (w == "f16")
      return FloatValue{Float16Value{(uint16_t)v}};
    if (w == "f32")
      return FloatValue{Float32Value{(float)v}};
    if (w == "f64")
      return FloatValue{Float64Value{(double)v}};

    Undefined();
  }

  // Special float literals printed type-first, e.g. `f32.inf`, `-f32.inf`,
  // `f32.nan`. The optional sign is part of the token text.
  FloatValue
  VisitPrimValueFloatSpecial(FutharkParser::PrimValueFloatSpecialContext *ctx) {
    auto text = ctx->SPECIAL_FLOAT()->getText(); // e.g. "-f32.inf"
    double mag = text.find("nan") != std::string::npos
                     ? std::numeric_limits<double>::quiet_NaN()
                     : std::numeric_limits<double>::infinity();
    double v = text.front() == '-' ? -mag : mag;
    if (text.find("f16") != std::string::npos)
      return FloatValue{Float16Value{(uint16_t)v}};
    if (text.find("f32") != std::string::npos)
      return FloatValue{Float32Value{(float)v}};
    if (text.find("f64") != std::string::npos)
      return FloatValue{Float64Value{(double)v}};

    Undefined();
  }

  std::string VisitId(antlr4::tree::TerminalNode *ctx) {
    return ctx->getText();
  }
};

inline Prog ParseFuthark(std::ifstream &file) {
  ANTLRInputStream input(file);
  FutharkLexer lexer(&input);
  CommonTokenStream tokens(&lexer);
  tokens.fill();
  FutharkParser parser(&tokens);
  auto tree = parser.root();

  FutharkTranslationVisitor visitor;
  return visitor.VisitProg(tree);
}
