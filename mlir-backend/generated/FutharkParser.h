
// Generated from C:/Users/cqs0520/source/repos/mocha-backend/rewrite/Futhark.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"




class  FutharkParser : public antlr4::Parser {
public:
  enum {
    T__0 = 1, T__1 = 2, T__2 = 3, T__3 = 4, T__4 = 5, T__5 = 6, T__6 = 7, 
    T__7 = 8, T__8 = 9, T__9 = 10, T__10 = 11, T__11 = 12, T__12 = 13, T__13 = 14, 
    T__14 = 15, T__15 = 16, T__16 = 17, T__17 = 18, T__18 = 19, T__19 = 20, 
    T__20 = 21, T__21 = 22, T__22 = 23, T__23 = 24, INTEGER_TYPE = 25, FLOAT_TYPE = 26, 
    NUMBER = 27, IOTA = 28, CONCAT = 29, ADD = 30, FADD = 31, SUB = 32, 
    FSUB = 33, MUL = 34, FMUL = 35, UDIV = 36, UDIVUP = 37, SDIV = 38, SDIVUP = 39, 
    FDIV = 40, FMOD = 41, UMOD = 42, SMOD = 43, SQUOT = 44, SREM = 45, SMIN = 46, 
    UMIN = 47, FMIN = 48, SMAX = 49, UMAX = 50, FMAX = 51, SHL = 52, LSHR = 53, 
    ASHR = 54, AND = 55, OR = 56, XOR = 57, POW = 58, FPOW = 59, LOGAND = 60, 
    LOGOR = 61, FLOAT = 62, STRING_LITERAL = 63, ID = 64, LPARAM = 65, RPARAM = 66, 
    OPEN_BRACKET = 67, CLOSE_BRACKET = 68, COMMA = 69, BUILTIN = 70, RETURN = 71, 
    FUNCTIONS = 72, RULES = 73, WS = 74
  };

  enum {
    RuleRoot = 0, RuleHeader = 1, RulePFunDef = 2, RulePEntry = 3, RulePEntryPointInput = 4, 
    RuleFParam = 5, RulePRetTypes = 6, RulePType = 7, RulePTypes = 8, RulePRetType = 9, 
    RulePExtShape = 10, RulePBody = 11, RulePStm = 12, RulePPat = 13, RulePPatElem = 14, 
    RulePExp = 15, RulePSoacOp = 16, RulePScrema = 17, RulePMapForm = 18, 
    RulePRedomapForm = 19, RulePReduce = 20, RulePComm = 21, RulePLambda = 22, 
    RulePLParam = 23, RulePApply = 24, RulePArg = 25, RulePArrayLit = 26, 
    RulePBasicOp = 27, RulePBinOp = 28, RuleBinaryOpcode = 29, RulePConvOp = 30, 
    RulePSubExp = 31, RulePPrimValue = 32, RulePPrimType = 33
  };

  explicit FutharkParser(antlr4::TokenStream *input);

  FutharkParser(antlr4::TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options);

  ~FutharkParser() override;

  std::string getGrammarFileName() const override;

  const antlr4::atn::ATN& getATN() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;


  class RootContext;
  class HeaderContext;
  class PFunDefContext;
  class PEntryContext;
  class PEntryPointInputContext;
  class FParamContext;
  class PRetTypesContext;
  class PTypeContext;
  class PTypesContext;
  class PRetTypeContext;
  class PExtShapeContext;
  class PBodyContext;
  class PStmContext;
  class PPatContext;
  class PPatElemContext;
  class PExpContext;
  class PSoacOpContext;
  class PScremaContext;
  class PMapFormContext;
  class PRedomapFormContext;
  class PReduceContext;
  class PCommContext;
  class PLambdaContext;
  class PLParamContext;
  class PApplyContext;
  class PArgContext;
  class PArrayLitContext;
  class PBasicOpContext;
  class PBinOpContext;
  class BinaryOpcodeContext;
  class PConvOpContext;
  class PSubExpContext;
  class PPrimValueContext;
  class PPrimTypeContext; 

  class  RootContext : public antlr4::ParserRuleContext {
  public:
    RootContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *EOF();
    HeaderContext *header();
    std::vector<PFunDefContext *> pFunDef();
    PFunDefContext* pFunDef(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  RootContext* root();

  class  HeaderContext : public antlr4::ParserRuleContext {
  public:
    HeaderContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> OPEN_BRACKET();
    antlr4::tree::TerminalNode* OPEN_BRACKET(size_t i);
    antlr4::tree::TerminalNode *NUMBER();
    std::vector<antlr4::tree::TerminalNode *> CLOSE_BRACKET();
    antlr4::tree::TerminalNode* CLOSE_BRACKET(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  HeaderContext* header();

  class  PFunDefContext : public antlr4::ParserRuleContext {
  public:
    PFunDefContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();
    antlr4::tree::TerminalNode *LPARAM();
    antlr4::tree::TerminalNode *RPARAM();
    std::vector<antlr4::tree::TerminalNode *> OPEN_BRACKET();
    antlr4::tree::TerminalNode* OPEN_BRACKET(size_t i);
    PTypeContext *pType();
    std::vector<antlr4::tree::TerminalNode *> CLOSE_BRACKET();
    antlr4::tree::TerminalNode* CLOSE_BRACKET(size_t i);
    PBodyContext *pBody();
    std::vector<FParamContext *> fParam();
    FParamContext* fParam(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    PEntryContext *pEntry();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PFunDefContext* pFunDef();

  class  PEntryContext : public antlr4::ParserRuleContext {
  public:
    PEntryContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPARAM();
    antlr4::tree::TerminalNode *STRING_LITERAL();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    std::vector<antlr4::tree::TerminalNode *> OPEN_BRACKET();
    antlr4::tree::TerminalNode* OPEN_BRACKET(size_t i);
    std::vector<antlr4::tree::TerminalNode *> CLOSE_BRACKET();
    antlr4::tree::TerminalNode* CLOSE_BRACKET(size_t i);
    PTypeContext *pType();
    antlr4::tree::TerminalNode *RPARAM();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PEntryContext* pEntry();

  class  PEntryPointInputContext : public antlr4::ParserRuleContext {
  public:
    PEntryPointInputContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PEntryPointInputContext* pEntryPointInput();

  class  FParamContext : public antlr4::ParserRuleContext {
  public:
    FParamContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();
    PTypeContext *pType();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  FParamContext* fParam();

  class  PRetTypesContext : public antlr4::ParserRuleContext {
  public:
    PRetTypesContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *OPEN_BRACKET();
    antlr4::tree::TerminalNode *CLOSE_BRACKET();
    std::vector<PTypeContext *> pType();
    PTypeContext* pType(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PRetTypesContext* pRetTypes();

  class  PTypeContext : public antlr4::ParserRuleContext {
  public:
    PTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    PTypeContext() = default;
    void copyFrom(PTypeContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  TypePrimContext : public PTypeContext {
  public:
    TypePrimContext(PTypeContext *ctx);

    PPrimTypeContext *pPrimType();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  TypeShapeContext : public PTypeContext {
  public:
    TypeShapeContext(PTypeContext *ctx);

    PExtShapeContext *pExtShape();
    PPrimTypeContext *pPrimType();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  PTypeContext* pType();

  class  PTypesContext : public antlr4::ParserRuleContext {
  public:
    PTypesContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *OPEN_BRACKET();
    antlr4::tree::TerminalNode *CLOSE_BRACKET();
    std::vector<PTypeContext *> pType();
    PTypeContext* pType(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PTypesContext* pTypes();

  class  PRetTypeContext : public antlr4::ParserRuleContext {
  public:
    PRetTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PTypeContext *pType();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PRetTypeContext* pRetType();

  class  PExtShapeContext : public antlr4::ParserRuleContext {
  public:
    PExtShapeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<PSubExpContext *> pSubExp();
    PSubExpContext* pSubExp(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PExtShapeContext* pExtShape();

  class  PBodyContext : public antlr4::ParserRuleContext {
  public:
    PBodyContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    PBodyContext() = default;
    void copyFrom(PBodyContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  EmptyBodyContext : public PBodyContext {
  public:
    EmptyBodyContext(PBodyContext *ctx);

    antlr4::tree::TerminalNode *OPEN_BRACKET();
    std::vector<PSubExpContext *> pSubExp();
    PSubExpContext* pSubExp(size_t i);
    antlr4::tree::TerminalNode *CLOSE_BRACKET();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  BodyContext : public PBodyContext {
  public:
    BodyContext(PBodyContext *ctx);

    antlr4::tree::TerminalNode *OPEN_BRACKET();
    std::vector<PSubExpContext *> pSubExp();
    PSubExpContext* pSubExp(size_t i);
    antlr4::tree::TerminalNode *CLOSE_BRACKET();
    std::vector<PStmContext *> pStm();
    PStmContext* pStm(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  PBodyContext* pBody();

  class  PStmContext : public antlr4::ParserRuleContext {
  public:
    PStmContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PPatContext *pPat();
    PExpContext *pExp();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PStmContext* pStm();

  class  PPatContext : public antlr4::ParserRuleContext {
  public:
    PPatContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *OPEN_BRACKET();
    std::vector<PPatElemContext *> pPatElem();
    PPatElemContext* pPatElem(size_t i);
    antlr4::tree::TerminalNode *CLOSE_BRACKET();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PPatContext* pPat();

  class  PPatElemContext : public antlr4::ParserRuleContext {
  public:
    PPatElemContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();
    PTypeContext *pType();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PPatElemContext* pPatElem();

  class  PExpContext : public antlr4::ParserRuleContext {
  public:
    PExpContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    PExpContext() = default;
    void copyFrom(PExpContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  ExpSubExpContext : public PExpContext {
  public:
    ExpSubExpContext(PExpContext *ctx);

    PSubExpContext *pSubExp();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  ExpApplyContext : public PExpContext {
  public:
    ExpApplyContext(PExpContext *ctx);

    PApplyContext *pApply();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  ExpBasicOpContext : public PExpContext {
  public:
    ExpBasicOpContext(PExpContext *ctx);

    PBasicOpContext *pBasicOp();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  ExpSoacOpContext : public PExpContext {
  public:
    ExpSoacOpContext(PExpContext *ctx);

    PSoacOpContext *pSoacOp();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  PExpContext* pExp();

  class  PSoacOpContext : public antlr4::ParserRuleContext {
  public:
    PSoacOpContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    PSoacOpContext() = default;
    void copyFrom(PSoacOpContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  SoacOpRedomapContext : public PSoacOpContext {
  public:
    SoacOpRedomapContext(PSoacOpContext *ctx);

    antlr4::tree::TerminalNode *LPARAM();
    PScremaContext *pScrema();
    PRedomapFormContext *pRedomapForm();
    antlr4::tree::TerminalNode *RPARAM();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  SoacOpMapContext : public PSoacOpContext {
  public:
    SoacOpMapContext(PSoacOpContext *ctx);

    antlr4::tree::TerminalNode *LPARAM();
    PScremaContext *pScrema();
    PMapFormContext *pMapForm();
    antlr4::tree::TerminalNode *RPARAM();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  PSoacOpContext* pSoacOp();

  class  PScremaContext : public antlr4::ParserRuleContext {
  public:
    PScremaContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PSubExpContext *pSubExp();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *OPEN_BRACKET();
    antlr4::tree::TerminalNode *CLOSE_BRACKET();
    std::vector<antlr4::tree::TerminalNode *> ID();
    antlr4::tree::TerminalNode* ID(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PScremaContext* pScrema();

  class  PMapFormContext : public antlr4::ParserRuleContext {
  public:
    PMapFormContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PLambdaContext *pLambda();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PMapFormContext* pMapForm();

  class  PRedomapFormContext : public antlr4::ParserRuleContext {
  public:
    PRedomapFormContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PLambdaContext *pLambda();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *OPEN_BRACKET();
    antlr4::tree::TerminalNode *CLOSE_BRACKET();
    std::vector<PReduceContext *> pReduce();
    PReduceContext* pReduce(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PRedomapFormContext* pRedomapForm();

  class  PReduceContext : public antlr4::ParserRuleContext {
  public:
    PReduceContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PCommContext *pComm();
    PLambdaContext *pLambda();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *OPEN_BRACKET();
    antlr4::tree::TerminalNode *CLOSE_BRACKET();
    std::vector<PSubExpContext *> pSubExp();
    PSubExpContext* pSubExp(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PReduceContext* pReduce();

  class  PCommContext : public antlr4::ParserRuleContext {
  public:
    PCommContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PCommContext* pComm();

  class  PLambdaContext : public antlr4::ParserRuleContext {
  public:
    PLambdaContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *OPEN_BRACKET();
    antlr4::tree::TerminalNode *CLOSE_BRACKET();
    PTypesContext *pTypes();
    PBodyContext *pBody();
    std::vector<PLParamContext *> pLParam();
    PLParamContext* pLParam(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PLambdaContext* pLambda();

  class  PLParamContext : public antlr4::ParserRuleContext {
  public:
    PLParamContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();
    PTypeContext *pType();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PLParamContext* pLParam();

  class  PApplyContext : public antlr4::ParserRuleContext {
  public:
    PApplyContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();
    antlr4::tree::TerminalNode *LPARAM();
    antlr4::tree::TerminalNode *RPARAM();
    PRetTypesContext *pRetTypes();
    std::vector<PSubExpContext *> pSubExp();
    PSubExpContext* pSubExp(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PApplyContext* pApply();

  class  PArgContext : public antlr4::ParserRuleContext {
  public:
    PArgContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PSubExpContext *pSubExp();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PArgContext* pArg();

  class  PArrayLitContext : public antlr4::ParserRuleContext {
  public:
    PArrayLitContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<PSubExpContext *> pSubExp();
    PSubExpContext* pSubExp(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PArrayLitContext* pArrayLit();

  class  PBasicOpContext : public antlr4::ParserRuleContext {
  public:
    PBasicOpContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    PBasicOpContext() = default;
    void copyFrom(PBasicOpContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  BasicOpArrayLitContext : public PBasicOpContext {
  public:
    BasicOpArrayLitContext(PBasicOpContext *ctx);

    PArrayLitContext *pArrayLit();
    PTypeContext *pType();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  BasicOpConcatContext : public PBasicOpContext {
  public:
    BasicOpConcatContext(PBasicOpContext *ctx);

    antlr4::tree::TerminalNode *CONCAT();
    antlr4::tree::TerminalNode *LPARAM();
    std::vector<PSubExpContext *> pSubExp();
    PSubExpContext* pSubExp(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *RPARAM();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  BasicOpConvContext : public PBasicOpContext {
  public:
    BasicOpConvContext(PBasicOpContext *ctx);

    PConvOpContext *pConvOp();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  BasicOpReplicateContext : public PBasicOpContext {
  public:
    BasicOpReplicateContext(PBasicOpContext *ctx);

    antlr4::tree::TerminalNode *LPARAM();
    PExtShapeContext *pExtShape();
    antlr4::tree::TerminalNode *COMMA();
    PSubExpContext *pSubExp();
    antlr4::tree::TerminalNode *RPARAM();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  BasicOpIotaContext : public PBasicOpContext {
  public:
    BasicOpIotaContext(PBasicOpContext *ctx);

    antlr4::tree::TerminalNode *IOTA();
    antlr4::tree::TerminalNode *LPARAM();
    std::vector<PSubExpContext *> pSubExp();
    PSubExpContext* pSubExp(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *RPARAM();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  BasicOpRearrangeContext : public PBasicOpContext {
  public:
    BasicOpRearrangeContext(PBasicOpContext *ctx);

    std::vector<antlr4::tree::TerminalNode *> LPARAM();
    antlr4::tree::TerminalNode* LPARAM(size_t i);
    PSubExpContext *pSubExp();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    std::vector<antlr4::tree::TerminalNode *> NUMBER();
    antlr4::tree::TerminalNode* NUMBER(size_t i);
    std::vector<antlr4::tree::TerminalNode *> RPARAM();
    antlr4::tree::TerminalNode* RPARAM(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  BasicOpFlatIndexContext : public PBasicOpContext {
  public:
    BasicOpFlatIndexContext(PBasicOpContext *ctx);

    std::vector<PSubExpContext *> pSubExp();
    PSubExpContext* pSubExp(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  BasicOpBinaryContext : public PBasicOpContext {
  public:
    BasicOpBinaryContext(PBasicOpContext *ctx);

    PBinOpContext *pBinOp();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  BasicOpReshapeContext : public PBasicOpContext {
  public:
    BasicOpReshapeContext(PBasicOpContext *ctx);

    std::vector<antlr4::tree::TerminalNode *> LPARAM();
    antlr4::tree::TerminalNode* LPARAM(size_t i);
    PSubExpContext *pSubExp();
    antlr4::tree::TerminalNode *COMMA();
    std::vector<PExtShapeContext *> pExtShape();
    PExtShapeContext* pExtShape(size_t i);
    std::vector<antlr4::tree::TerminalNode *> RPARAM();
    antlr4::tree::TerminalNode* RPARAM(size_t i);
    std::vector<antlr4::tree::TerminalNode *> NUMBER();
    antlr4::tree::TerminalNode* NUMBER(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  PBasicOpContext* pBasicOp();

  class  PBinOpContext : public antlr4::ParserRuleContext {
  public:
    PBinOpContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    BinaryOpcodeContext *binaryOpcode();
    antlr4::tree::TerminalNode *LPARAM();
    std::vector<PSubExpContext *> pSubExp();
    PSubExpContext* pSubExp(size_t i);
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *RPARAM();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PBinOpContext* pBinOp();

  class  BinaryOpcodeContext : public antlr4::ParserRuleContext {
  public:
    BinaryOpcodeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ADD();
    antlr4::tree::TerminalNode *FADD();
    antlr4::tree::TerminalNode *SUB();
    antlr4::tree::TerminalNode *FSUB();
    antlr4::tree::TerminalNode *MUL();
    antlr4::tree::TerminalNode *FMUL();
    antlr4::tree::TerminalNode *UDIV();
    antlr4::tree::TerminalNode *UDIVUP();
    antlr4::tree::TerminalNode *SDIV();
    antlr4::tree::TerminalNode *SDIVUP();
    antlr4::tree::TerminalNode *FDIV();
    antlr4::tree::TerminalNode *FMOD();
    antlr4::tree::TerminalNode *UMOD();
    antlr4::tree::TerminalNode *SMOD();
    antlr4::tree::TerminalNode *SQUOT();
    antlr4::tree::TerminalNode *SREM();
    antlr4::tree::TerminalNode *SMIN();
    antlr4::tree::TerminalNode *UMIN();
    antlr4::tree::TerminalNode *FMIN();
    antlr4::tree::TerminalNode *SMAX();
    antlr4::tree::TerminalNode *UMAX();
    antlr4::tree::TerminalNode *FMAX();
    antlr4::tree::TerminalNode *SHL();
    antlr4::tree::TerminalNode *LSHR();
    antlr4::tree::TerminalNode *ASHR();
    antlr4::tree::TerminalNode *AND();
    antlr4::tree::TerminalNode *OR();
    antlr4::tree::TerminalNode *XOR();
    antlr4::tree::TerminalNode *POW();
    antlr4::tree::TerminalNode *FPOW();
    antlr4::tree::TerminalNode *LOGAND();
    antlr4::tree::TerminalNode *LOGOR();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  BinaryOpcodeContext* binaryOpcode();

  class  PConvOpContext : public antlr4::ParserRuleContext {
  public:
    PConvOpContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<PPrimTypeContext *> pPrimType();
    PPrimTypeContext* pPrimType(size_t i);
    PSubExpContext *pSubExp();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PConvOpContext* pConvOp();

  class  PSubExpContext : public antlr4::ParserRuleContext {
  public:
    PSubExpContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    PSubExpContext() = default;
    void copyFrom(PSubExpContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  VarSubExpContext : public PSubExpContext {
  public:
    VarSubExpContext(PSubExpContext *ctx);

    antlr4::tree::TerminalNode *ID();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  ConstantSubExpressionContext : public PSubExpContext {
  public:
    ConstantSubExpressionContext(PSubExpContext *ctx);

    PPrimValueContext *pPrimValue();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  PSubExpContext* pSubExp();

  class  PPrimValueContext : public antlr4::ParserRuleContext {
  public:
    PPrimValueContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    PPrimValueContext() = default;
    void copyFrom(PPrimValueContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  PrimValueIntegerContext : public PPrimValueContext {
  public:
    PrimValueIntegerContext(PPrimValueContext *ctx);

    antlr4::tree::TerminalNode *NUMBER();
    antlr4::tree::TerminalNode *INTEGER_TYPE();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  PrimValueFloatContext : public PPrimValueContext {
  public:
    PrimValueFloatContext(PPrimValueContext *ctx);

    antlr4::tree::TerminalNode *FLOAT();
    antlr4::tree::TerminalNode *FLOAT_TYPE();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  PPrimValueContext* pPrimValue();

  class  PPrimTypeContext : public antlr4::ParserRuleContext {
  public:
    PPrimTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    PPrimTypeContext() = default;
    void copyFrom(PPrimTypeContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  PrimTypeIntegerContext : public PPrimTypeContext {
  public:
    PrimTypeIntegerContext(PPrimTypeContext *ctx);

    antlr4::tree::TerminalNode *INTEGER_TYPE();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class  PrimTypeFloatContext : public PPrimTypeContext {
  public:
    PrimTypeFloatContext(PPrimTypeContext *ctx);

    antlr4::tree::TerminalNode *FLOAT_TYPE();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  PPrimTypeContext* pPrimType();


  // By default the static state used to implement the parser is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:
};

