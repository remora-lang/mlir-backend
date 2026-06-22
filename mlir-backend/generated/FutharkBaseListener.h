
// Generated from C:/Users/cqs0520/source/repos/mocha-backend/rewrite/Futhark.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "FutharkListener.h"


/**
 * This class provides an empty implementation of FutharkListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  FutharkBaseListener : public FutharkListener {
public:

  virtual void enterRoot(FutharkParser::RootContext * /*ctx*/) override { }
  virtual void exitRoot(FutharkParser::RootContext * /*ctx*/) override { }

  virtual void enterHeader(FutharkParser::HeaderContext * /*ctx*/) override { }
  virtual void exitHeader(FutharkParser::HeaderContext * /*ctx*/) override { }

  virtual void enterPFunDef(FutharkParser::PFunDefContext * /*ctx*/) override { }
  virtual void exitPFunDef(FutharkParser::PFunDefContext * /*ctx*/) override { }

  virtual void enterPEntry(FutharkParser::PEntryContext * /*ctx*/) override { }
  virtual void exitPEntry(FutharkParser::PEntryContext * /*ctx*/) override { }

  virtual void enterPEntryPointInput(FutharkParser::PEntryPointInputContext * /*ctx*/) override { }
  virtual void exitPEntryPointInput(FutharkParser::PEntryPointInputContext * /*ctx*/) override { }

  virtual void enterFParam(FutharkParser::FParamContext * /*ctx*/) override { }
  virtual void exitFParam(FutharkParser::FParamContext * /*ctx*/) override { }

  virtual void enterPRetTypes(FutharkParser::PRetTypesContext * /*ctx*/) override { }
  virtual void exitPRetTypes(FutharkParser::PRetTypesContext * /*ctx*/) override { }

  virtual void enterTypeShape(FutharkParser::TypeShapeContext * /*ctx*/) override { }
  virtual void exitTypeShape(FutharkParser::TypeShapeContext * /*ctx*/) override { }

  virtual void enterTypePrim(FutharkParser::TypePrimContext * /*ctx*/) override { }
  virtual void exitTypePrim(FutharkParser::TypePrimContext * /*ctx*/) override { }

  virtual void enterPTypes(FutharkParser::PTypesContext * /*ctx*/) override { }
  virtual void exitPTypes(FutharkParser::PTypesContext * /*ctx*/) override { }

  virtual void enterPRetType(FutharkParser::PRetTypeContext * /*ctx*/) override { }
  virtual void exitPRetType(FutharkParser::PRetTypeContext * /*ctx*/) override { }

  virtual void enterPExtShape(FutharkParser::PExtShapeContext * /*ctx*/) override { }
  virtual void exitPExtShape(FutharkParser::PExtShapeContext * /*ctx*/) override { }

  virtual void enterBody(FutharkParser::BodyContext * /*ctx*/) override { }
  virtual void exitBody(FutharkParser::BodyContext * /*ctx*/) override { }

  virtual void enterEmptyBody(FutharkParser::EmptyBodyContext * /*ctx*/) override { }
  virtual void exitEmptyBody(FutharkParser::EmptyBodyContext * /*ctx*/) override { }

  virtual void enterPStm(FutharkParser::PStmContext * /*ctx*/) override { }
  virtual void exitPStm(FutharkParser::PStmContext * /*ctx*/) override { }

  virtual void enterPPat(FutharkParser::PPatContext * /*ctx*/) override { }
  virtual void exitPPat(FutharkParser::PPatContext * /*ctx*/) override { }

  virtual void enterPPatElem(FutharkParser::PPatElemContext * /*ctx*/) override { }
  virtual void exitPPatElem(FutharkParser::PPatElemContext * /*ctx*/) override { }

  virtual void enterExpApply(FutharkParser::ExpApplyContext * /*ctx*/) override { }
  virtual void exitExpApply(FutharkParser::ExpApplyContext * /*ctx*/) override { }

  virtual void enterExpBasicOp(FutharkParser::ExpBasicOpContext * /*ctx*/) override { }
  virtual void exitExpBasicOp(FutharkParser::ExpBasicOpContext * /*ctx*/) override { }

  virtual void enterExpSoacOp(FutharkParser::ExpSoacOpContext * /*ctx*/) override { }
  virtual void exitExpSoacOp(FutharkParser::ExpSoacOpContext * /*ctx*/) override { }

  virtual void enterExpSubExp(FutharkParser::ExpSubExpContext * /*ctx*/) override { }
  virtual void exitExpSubExp(FutharkParser::ExpSubExpContext * /*ctx*/) override { }

  virtual void enterSoacOpMap(FutharkParser::SoacOpMapContext * /*ctx*/) override { }
  virtual void exitSoacOpMap(FutharkParser::SoacOpMapContext * /*ctx*/) override { }

  virtual void enterSoacOpRedomap(FutharkParser::SoacOpRedomapContext * /*ctx*/) override { }
  virtual void exitSoacOpRedomap(FutharkParser::SoacOpRedomapContext * /*ctx*/) override { }

  virtual void enterPScrema(FutharkParser::PScremaContext * /*ctx*/) override { }
  virtual void exitPScrema(FutharkParser::PScremaContext * /*ctx*/) override { }

  virtual void enterPMapForm(FutharkParser::PMapFormContext * /*ctx*/) override { }
  virtual void exitPMapForm(FutharkParser::PMapFormContext * /*ctx*/) override { }

  virtual void enterPRedomapForm(FutharkParser::PRedomapFormContext * /*ctx*/) override { }
  virtual void exitPRedomapForm(FutharkParser::PRedomapFormContext * /*ctx*/) override { }

  virtual void enterPReduce(FutharkParser::PReduceContext * /*ctx*/) override { }
  virtual void exitPReduce(FutharkParser::PReduceContext * /*ctx*/) override { }

  virtual void enterPComm(FutharkParser::PCommContext * /*ctx*/) override { }
  virtual void exitPComm(FutharkParser::PCommContext * /*ctx*/) override { }

  virtual void enterPLambda(FutharkParser::PLambdaContext * /*ctx*/) override { }
  virtual void exitPLambda(FutharkParser::PLambdaContext * /*ctx*/) override { }

  virtual void enterPLParam(FutharkParser::PLParamContext * /*ctx*/) override { }
  virtual void exitPLParam(FutharkParser::PLParamContext * /*ctx*/) override { }

  virtual void enterPApply(FutharkParser::PApplyContext * /*ctx*/) override { }
  virtual void exitPApply(FutharkParser::PApplyContext * /*ctx*/) override { }

  virtual void enterPArg(FutharkParser::PArgContext * /*ctx*/) override { }
  virtual void exitPArg(FutharkParser::PArgContext * /*ctx*/) override { }

  virtual void enterPArrayLit(FutharkParser::PArrayLitContext * /*ctx*/) override { }
  virtual void exitPArrayLit(FutharkParser::PArrayLitContext * /*ctx*/) override { }

  virtual void enterBasicOpReplicate(FutharkParser::BasicOpReplicateContext * /*ctx*/) override { }
  virtual void exitBasicOpReplicate(FutharkParser::BasicOpReplicateContext * /*ctx*/) override { }

  virtual void enterBasicOpArrayLit(FutharkParser::BasicOpArrayLitContext * /*ctx*/) override { }
  virtual void exitBasicOpArrayLit(FutharkParser::BasicOpArrayLitContext * /*ctx*/) override { }

  virtual void enterBasicOpFlatIndex(FutharkParser::BasicOpFlatIndexContext * /*ctx*/) override { }
  virtual void exitBasicOpFlatIndex(FutharkParser::BasicOpFlatIndexContext * /*ctx*/) override { }

  virtual void enterBasicOpConcat(FutharkParser::BasicOpConcatContext * /*ctx*/) override { }
  virtual void exitBasicOpConcat(FutharkParser::BasicOpConcatContext * /*ctx*/) override { }

  virtual void enterBasicOpIota(FutharkParser::BasicOpIotaContext * /*ctx*/) override { }
  virtual void exitBasicOpIota(FutharkParser::BasicOpIotaContext * /*ctx*/) override { }

  virtual void enterBasicOpBinary(FutharkParser::BasicOpBinaryContext * /*ctx*/) override { }
  virtual void exitBasicOpBinary(FutharkParser::BasicOpBinaryContext * /*ctx*/) override { }

  virtual void enterBasicOpConv(FutharkParser::BasicOpConvContext * /*ctx*/) override { }
  virtual void exitBasicOpConv(FutharkParser::BasicOpConvContext * /*ctx*/) override { }

  virtual void enterBasicOpReshape(FutharkParser::BasicOpReshapeContext * /*ctx*/) override { }
  virtual void exitBasicOpReshape(FutharkParser::BasicOpReshapeContext * /*ctx*/) override { }

  virtual void enterBasicOpRearrange(FutharkParser::BasicOpRearrangeContext * /*ctx*/) override { }
  virtual void exitBasicOpRearrange(FutharkParser::BasicOpRearrangeContext * /*ctx*/) override { }

  virtual void enterPBinOp(FutharkParser::PBinOpContext * /*ctx*/) override { }
  virtual void exitPBinOp(FutharkParser::PBinOpContext * /*ctx*/) override { }

  virtual void enterBinaryOpcode(FutharkParser::BinaryOpcodeContext * /*ctx*/) override { }
  virtual void exitBinaryOpcode(FutharkParser::BinaryOpcodeContext * /*ctx*/) override { }

  virtual void enterPConvOp(FutharkParser::PConvOpContext * /*ctx*/) override { }
  virtual void exitPConvOp(FutharkParser::PConvOpContext * /*ctx*/) override { }

  virtual void enterConstantSubExpression(FutharkParser::ConstantSubExpressionContext * /*ctx*/) override { }
  virtual void exitConstantSubExpression(FutharkParser::ConstantSubExpressionContext * /*ctx*/) override { }

  virtual void enterVarSubExp(FutharkParser::VarSubExpContext * /*ctx*/) override { }
  virtual void exitVarSubExp(FutharkParser::VarSubExpContext * /*ctx*/) override { }

  virtual void enterPrimValueInteger(FutharkParser::PrimValueIntegerContext * /*ctx*/) override { }
  virtual void exitPrimValueInteger(FutharkParser::PrimValueIntegerContext * /*ctx*/) override { }

  virtual void enterPrimValueFloat(FutharkParser::PrimValueFloatContext * /*ctx*/) override { }
  virtual void exitPrimValueFloat(FutharkParser::PrimValueFloatContext * /*ctx*/) override { }

  virtual void enterPrimTypeInteger(FutharkParser::PrimTypeIntegerContext * /*ctx*/) override { }
  virtual void exitPrimTypeInteger(FutharkParser::PrimTypeIntegerContext * /*ctx*/) override { }

  virtual void enterPrimTypeFloat(FutharkParser::PrimTypeFloatContext * /*ctx*/) override { }
  virtual void exitPrimTypeFloat(FutharkParser::PrimTypeFloatContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

