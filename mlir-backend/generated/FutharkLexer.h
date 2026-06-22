
// Generated from C:/Users/cqs0520/source/repos/mocha-backend/rewrite/Futhark.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"




class  FutharkLexer : public antlr4::Lexer {
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

  explicit FutharkLexer(antlr4::CharStream *input);

  ~FutharkLexer() override;


  std::string getGrammarFileName() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const std::vector<std::string>& getChannelNames() const override;

  const std::vector<std::string>& getModeNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;

  const antlr4::atn::ATN& getATN() const override;

  // By default the static state used to implement the lexer is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:

  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

};

