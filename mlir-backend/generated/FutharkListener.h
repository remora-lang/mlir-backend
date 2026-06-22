
// Generated from C:/Users/cqs0520/source/repos/mocha-backend/rewrite/Futhark.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "FutharkParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by FutharkParser.
 */
class  FutharkListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterRoot(FutharkParser::RootContext *ctx) = 0;
  virtual void exitRoot(FutharkParser::RootContext *ctx) = 0;

  virtual void enterHeader(FutharkParser::HeaderContext *ctx) = 0;
  virtual void exitHeader(FutharkParser::HeaderContext *ctx) = 0;

  virtual void enterPFunDef(FutharkParser::PFunDefContext *ctx) = 0;
  virtual void exitPFunDef(FutharkParser::PFunDefContext *ctx) = 0;

  virtual void enterPEntry(FutharkParser::PEntryContext *ctx) = 0;
  virtual void exitPEntry(FutharkParser::PEntryContext *ctx) = 0;

  virtual void enterPEntryPointInput(FutharkParser::PEntryPointInputContext *ctx) = 0;
  virtual void exitPEntryPointInput(FutharkParser::PEntryPointInputContext *ctx) = 0;

  virtual void enterFParam(FutharkParser::FParamContext *ctx) = 0;
  virtual void exitFParam(FutharkParser::FParamContext *ctx) = 0;

  virtual void enterPRetTypes(FutharkParser::PRetTypesContext *ctx) = 0;
  virtual void exitPRetTypes(FutharkParser::PRetTypesContext *ctx) = 0;

  virtual void enterTypeShape(FutharkParser::TypeShapeContext *ctx) = 0;
  virtual void exitTypeShape(FutharkParser::TypeShapeContext *ctx) = 0;

  virtual void enterTypePrim(FutharkParser::TypePrimContext *ctx) = 0;
  virtual void exitTypePrim(FutharkParser::TypePrimContext *ctx) = 0;

  virtual void enterPTypes(FutharkParser::PTypesContext *ctx) = 0;
  virtual void exitPTypes(FutharkParser::PTypesContext *ctx) = 0;

  virtual void enterPRetType(FutharkParser::PRetTypeContext *ctx) = 0;
  virtual void exitPRetType(FutharkParser::PRetTypeContext *ctx) = 0;

  virtual void enterPExtShape(FutharkParser::PExtShapeContext *ctx) = 0;
  virtual void exitPExtShape(FutharkParser::PExtShapeContext *ctx) = 0;

  virtual void enterBody(FutharkParser::BodyContext *ctx) = 0;
  virtual void exitBody(FutharkParser::BodyContext *ctx) = 0;

  virtual void enterEmptyBody(FutharkParser::EmptyBodyContext *ctx) = 0;
  virtual void exitEmptyBody(FutharkParser::EmptyBodyContext *ctx) = 0;

  virtual void enterPStm(FutharkParser::PStmContext *ctx) = 0;
  virtual void exitPStm(FutharkParser::PStmContext *ctx) = 0;

  virtual void enterPPat(FutharkParser::PPatContext *ctx) = 0;
  virtual void exitPPat(FutharkParser::PPatContext *ctx) = 0;

  virtual void enterPPatElem(FutharkParser::PPatElemContext *ctx) = 0;
  virtual void exitPPatElem(FutharkParser::PPatElemContext *ctx) = 0;

  virtual void enterExpApply(FutharkParser::ExpApplyContext *ctx) = 0;
  virtual void exitExpApply(FutharkParser::ExpApplyContext *ctx) = 0;

  virtual void enterExpBasicOp(FutharkParser::ExpBasicOpContext *ctx) = 0;
  virtual void exitExpBasicOp(FutharkParser::ExpBasicOpContext *ctx) = 0;

  virtual void enterExpSoacOp(FutharkParser::ExpSoacOpContext *ctx) = 0;
  virtual void exitExpSoacOp(FutharkParser::ExpSoacOpContext *ctx) = 0;

  virtual void enterExpSubExp(FutharkParser::ExpSubExpContext *ctx) = 0;
  virtual void exitExpSubExp(FutharkParser::ExpSubExpContext *ctx) = 0;

  virtual void enterSoacOpMap(FutharkParser::SoacOpMapContext *ctx) = 0;
  virtual void exitSoacOpMap(FutharkParser::SoacOpMapContext *ctx) = 0;

  virtual void enterSoacOpRedomap(FutharkParser::SoacOpRedomapContext *ctx) = 0;
  virtual void exitSoacOpRedomap(FutharkParser::SoacOpRedomapContext *ctx) = 0;

  virtual void enterPScrema(FutharkParser::PScremaContext *ctx) = 0;
  virtual void exitPScrema(FutharkParser::PScremaContext *ctx) = 0;

  virtual void enterPMapForm(FutharkParser::PMapFormContext *ctx) = 0;
  virtual void exitPMapForm(FutharkParser::PMapFormContext *ctx) = 0;

  virtual void enterPRedomapForm(FutharkParser::PRedomapFormContext *ctx) = 0;
  virtual void exitPRedomapForm(FutharkParser::PRedomapFormContext *ctx) = 0;

  virtual void enterPReduce(FutharkParser::PReduceContext *ctx) = 0;
  virtual void exitPReduce(FutharkParser::PReduceContext *ctx) = 0;

  virtual void enterPComm(FutharkParser::PCommContext *ctx) = 0;
  virtual void exitPComm(FutharkParser::PCommContext *ctx) = 0;

  virtual void enterPLambda(FutharkParser::PLambdaContext *ctx) = 0;
  virtual void exitPLambda(FutharkParser::PLambdaContext *ctx) = 0;

  virtual void enterPLParam(FutharkParser::PLParamContext *ctx) = 0;
  virtual void exitPLParam(FutharkParser::PLParamContext *ctx) = 0;

  virtual void enterPApply(FutharkParser::PApplyContext *ctx) = 0;
  virtual void exitPApply(FutharkParser::PApplyContext *ctx) = 0;

  virtual void enterPArg(FutharkParser::PArgContext *ctx) = 0;
  virtual void exitPArg(FutharkParser::PArgContext *ctx) = 0;

  virtual void enterPArrayLit(FutharkParser::PArrayLitContext *ctx) = 0;
  virtual void exitPArrayLit(FutharkParser::PArrayLitContext *ctx) = 0;

  virtual void enterBasicOpReplicate(FutharkParser::BasicOpReplicateContext *ctx) = 0;
  virtual void exitBasicOpReplicate(FutharkParser::BasicOpReplicateContext *ctx) = 0;

  virtual void enterBasicOpArrayLit(FutharkParser::BasicOpArrayLitContext *ctx) = 0;
  virtual void exitBasicOpArrayLit(FutharkParser::BasicOpArrayLitContext *ctx) = 0;

  virtual void enterBasicOpFlatIndex(FutharkParser::BasicOpFlatIndexContext *ctx) = 0;
  virtual void exitBasicOpFlatIndex(FutharkParser::BasicOpFlatIndexContext *ctx) = 0;

  virtual void enterBasicOpConcat(FutharkParser::BasicOpConcatContext *ctx) = 0;
  virtual void exitBasicOpConcat(FutharkParser::BasicOpConcatContext *ctx) = 0;

  virtual void enterBasicOpIota(FutharkParser::BasicOpIotaContext *ctx) = 0;
  virtual void exitBasicOpIota(FutharkParser::BasicOpIotaContext *ctx) = 0;

  virtual void enterBasicOpBinary(FutharkParser::BasicOpBinaryContext *ctx) = 0;
  virtual void exitBasicOpBinary(FutharkParser::BasicOpBinaryContext *ctx) = 0;

  virtual void enterBasicOpConv(FutharkParser::BasicOpConvContext *ctx) = 0;
  virtual void exitBasicOpConv(FutharkParser::BasicOpConvContext *ctx) = 0;

  virtual void enterBasicOpReshape(FutharkParser::BasicOpReshapeContext *ctx) = 0;
  virtual void exitBasicOpReshape(FutharkParser::BasicOpReshapeContext *ctx) = 0;

  virtual void enterBasicOpRearrange(FutharkParser::BasicOpRearrangeContext *ctx) = 0;
  virtual void exitBasicOpRearrange(FutharkParser::BasicOpRearrangeContext *ctx) = 0;

  virtual void enterPBinOp(FutharkParser::PBinOpContext *ctx) = 0;
  virtual void exitPBinOp(FutharkParser::PBinOpContext *ctx) = 0;

  virtual void enterBinaryOpcode(FutharkParser::BinaryOpcodeContext *ctx) = 0;
  virtual void exitBinaryOpcode(FutharkParser::BinaryOpcodeContext *ctx) = 0;

  virtual void enterPConvOp(FutharkParser::PConvOpContext *ctx) = 0;
  virtual void exitPConvOp(FutharkParser::PConvOpContext *ctx) = 0;

  virtual void enterConstantSubExpression(FutharkParser::ConstantSubExpressionContext *ctx) = 0;
  virtual void exitConstantSubExpression(FutharkParser::ConstantSubExpressionContext *ctx) = 0;

  virtual void enterVarSubExp(FutharkParser::VarSubExpContext *ctx) = 0;
  virtual void exitVarSubExp(FutharkParser::VarSubExpContext *ctx) = 0;

  virtual void enterPrimValueInteger(FutharkParser::PrimValueIntegerContext *ctx) = 0;
  virtual void exitPrimValueInteger(FutharkParser::PrimValueIntegerContext *ctx) = 0;

  virtual void enterPrimValueFloat(FutharkParser::PrimValueFloatContext *ctx) = 0;
  virtual void exitPrimValueFloat(FutharkParser::PrimValueFloatContext *ctx) = 0;

  virtual void enterPrimTypeInteger(FutharkParser::PrimTypeIntegerContext *ctx) = 0;
  virtual void exitPrimTypeInteger(FutharkParser::PrimTypeIntegerContext *ctx) = 0;

  virtual void enterPrimTypeFloat(FutharkParser::PrimTypeFloatContext *ctx) = 0;
  virtual void exitPrimTypeFloat(FutharkParser::PrimTypeFloatContext *ctx) = 0;


};

