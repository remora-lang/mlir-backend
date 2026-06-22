
// Generated from C:/Users/cqs0520/source/repos/mocha-backend/rewrite/Futhark.g4 by ANTLR 4.13.2


#include "FutharkListener.h"

#include "FutharkParser.h"


using namespace antlrcpp;

using namespace antlr4;

namespace {

struct FutharkParserStaticData final {
  FutharkParserStaticData(std::vector<std::string> ruleNames,
                        std::vector<std::string> literalNames,
                        std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  FutharkParserStaticData(const FutharkParserStaticData&) = delete;
  FutharkParserStaticData(FutharkParserStaticData&&) = delete;
  FutharkParserStaticData& operator=(const FutharkParserStaticData&) = delete;
  FutharkParserStaticData& operator=(FutharkParserStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

::antlr4::internal::OnceFlag futharkParserOnceFlag;
#if ANTLR4_USE_THREAD_LOCAL_CACHE
static thread_local
#endif
std::unique_ptr<FutharkParserStaticData> futharkParserStaticData = nullptr;

void futharkParserInitialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  if (futharkParserStaticData != nullptr) {
    return;
  }
#else
  assert(futharkParserStaticData == nullptr);
#endif
  auto staticData = std::make_unique<FutharkParserStaticData>(
    std::vector<std::string>{
      "root", "header", "pFunDef", "pEntry", "pEntryPointInput", "fParam", 
      "pRetTypes", "pType", "pTypes", "pRetType", "pExtShape", "pBody", 
      "pStm", "pPat", "pPatElem", "pExp", "pSoacOp", "pScrema", "pMapForm", 
      "pRedomapForm", "pReduce", "pComm", "pLambda", "pLParam", "pApply", 
      "pArg", "pArrayLit", "pBasicOp", "pBinOp", "binaryOpcode", "pConvOp", 
      "pSubExp", "pPrimValue", "pPrimType"
    },
    std::vector<std::string>{
      "", "'name_source'", "'types'", "'fun'", "':'", "'='", "'entry'", 
      "'['", "']'", "'in'", "'let'", "'map'", "'redomap'", "'commutative'", 
      "'\\'", "'->'", "'apply'", "'replicate'", "'reshape'", "'::'", "'=>'", 
      "';'", "'rearrange'", "'sext'", "'to'", "", "", "", "", "", "", "", 
      "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
      "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "'('", 
      "')'", "'{'", "'}'", "','", "'builtin'", "'return'", "'FUNCTIONS'", 
      "'RULES'"
    },
    std::vector<std::string>{
      "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
      "", "", "", "", "", "", "", "", "INTEGER_TYPE", "FLOAT_TYPE", "NUMBER", 
      "IOTA", "CONCAT", "ADD", "FADD", "SUB", "FSUB", "MUL", "FMUL", "UDIV", 
      "UDIVUP", "SDIV", "SDIVUP", "FDIV", "FMOD", "UMOD", "SMOD", "SQUOT", 
      "SREM", "SMIN", "UMIN", "FMIN", "SMAX", "UMAX", "FMAX", "SHL", "LSHR", 
      "ASHR", "AND", "OR", "XOR", "POW", "FPOW", "LOGAND", "LOGOR", "FLOAT", 
      "STRING_LITERAL", "ID", "LPARAM", "RPARAM", "OPEN_BRACKET", "CLOSE_BRACKET", 
      "COMMA", "BUILTIN", "RETURN", "FUNCTIONS", "RULES", "WS"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,1,74,464,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,2,
  	7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,2,14,7,
  	14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,7,20,2,21,7,
  	21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,2,27,7,27,2,28,7,
  	28,2,29,7,29,2,30,7,30,2,31,7,31,2,32,7,32,2,33,7,33,1,0,3,0,70,8,0,1,
  	0,4,0,73,8,0,11,0,12,0,74,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,
  	3,2,88,8,2,1,2,3,2,91,8,2,1,2,1,2,1,2,3,2,96,8,2,1,2,1,2,5,2,100,8,2,
  	10,2,12,2,103,9,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,3,1,3,1,3,
  	1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,4,1,4,1,5,1,5,1,5,1,5,1,6,1,6,5,
  	6,135,8,6,10,6,12,6,138,9,6,1,6,1,6,5,6,142,8,6,10,6,12,6,145,9,6,1,6,
  	1,6,1,7,1,7,1,7,1,7,3,7,153,8,7,1,8,1,8,3,8,157,8,8,1,8,1,8,5,8,161,8,
  	8,10,8,12,8,164,9,8,1,8,1,8,1,9,1,9,1,10,1,10,5,10,172,8,10,10,10,12,
  	10,175,9,10,1,10,4,10,178,8,10,11,10,12,10,179,1,11,5,11,183,8,11,10,
  	11,12,11,186,9,11,1,11,1,11,1,11,1,11,1,11,5,11,193,8,11,10,11,12,11,
  	196,9,11,1,11,1,11,1,11,1,11,1,11,1,11,5,11,204,8,11,10,11,12,11,207,
  	9,11,1,11,1,11,3,11,211,8,11,1,12,1,12,1,12,1,12,1,12,1,13,1,13,1,13,
  	1,13,1,13,5,13,223,8,13,10,13,12,13,226,9,13,1,13,1,13,1,14,1,14,1,14,
  	1,14,1,15,1,15,1,15,1,15,3,15,238,8,15,1,16,1,16,1,16,1,16,1,16,1,16,
  	1,16,1,16,1,16,1,16,1,16,1,16,3,16,252,8,16,1,17,1,17,1,17,1,17,3,17,
  	258,8,17,1,17,1,17,5,17,262,8,17,10,17,12,17,265,9,17,1,17,1,17,1,17,
  	1,18,1,18,1,19,1,19,1,19,1,19,3,19,276,8,19,1,19,1,19,5,19,280,8,19,10,
  	19,12,19,283,9,19,1,19,1,19,1,20,1,20,1,20,1,20,1,20,3,20,292,8,20,1,
  	20,1,20,5,20,296,8,20,10,20,12,20,299,9,20,1,20,1,20,1,21,3,21,304,8,
  	21,1,22,1,22,1,22,3,22,309,8,22,1,22,1,22,5,22,313,8,22,10,22,12,22,316,
  	9,22,1,22,1,22,1,22,1,22,1,22,1,22,1,23,1,23,1,23,1,23,1,24,1,24,1,24,
  	1,24,3,24,332,8,24,1,24,1,24,5,24,336,8,24,10,24,12,24,339,9,24,1,24,
  	1,24,1,24,1,24,1,25,1,25,1,26,1,26,1,26,1,26,5,26,351,8,26,10,26,12,26,
  	354,9,26,1,26,1,26,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,
  	1,27,1,27,1,27,1,27,1,27,1,27,5,27,374,8,27,10,27,12,27,377,9,27,1,27,
  	1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,
  	1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,
  	1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,
  	1,27,1,27,1,27,5,27,425,8,27,10,27,12,27,428,9,27,1,27,1,27,1,27,3,27,
  	433,8,27,1,28,1,28,1,28,1,28,1,28,1,28,1,28,1,29,1,29,1,30,1,30,1,30,
  	1,30,1,30,1,30,1,31,1,31,3,31,452,8,31,1,32,1,32,1,32,1,32,3,32,458,8,
  	32,1,33,1,33,3,33,462,8,33,1,33,0,0,34,0,2,4,6,8,10,12,14,16,18,20,22,
  	24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,0,1,
  	1,0,30,61,476,0,69,1,0,0,0,2,78,1,0,0,0,4,90,1,0,0,0,6,114,1,0,0,0,8,
  	126,1,0,0,0,10,128,1,0,0,0,12,132,1,0,0,0,14,152,1,0,0,0,16,154,1,0,0,
  	0,18,167,1,0,0,0,20,177,1,0,0,0,22,210,1,0,0,0,24,212,1,0,0,0,26,217,
  	1,0,0,0,28,229,1,0,0,0,30,237,1,0,0,0,32,251,1,0,0,0,34,253,1,0,0,0,36,
  	269,1,0,0,0,38,271,1,0,0,0,40,286,1,0,0,0,42,303,1,0,0,0,44,305,1,0,0,
  	0,46,323,1,0,0,0,48,327,1,0,0,0,50,344,1,0,0,0,52,346,1,0,0,0,54,432,
  	1,0,0,0,56,434,1,0,0,0,58,441,1,0,0,0,60,443,1,0,0,0,62,451,1,0,0,0,64,
  	457,1,0,0,0,66,461,1,0,0,0,68,70,3,2,1,0,69,68,1,0,0,0,69,70,1,0,0,0,
  	70,72,1,0,0,0,71,73,3,4,2,0,72,71,1,0,0,0,73,74,1,0,0,0,74,72,1,0,0,0,
  	74,75,1,0,0,0,75,76,1,0,0,0,76,77,5,0,0,1,77,1,1,0,0,0,78,79,5,1,0,0,
  	79,80,5,67,0,0,80,81,5,27,0,0,81,82,5,68,0,0,82,83,5,2,0,0,83,84,5,67,
  	0,0,84,85,5,68,0,0,85,3,1,0,0,0,86,88,3,6,3,0,87,86,1,0,0,0,87,88,1,0,
  	0,0,88,91,1,0,0,0,89,91,5,3,0,0,90,87,1,0,0,0,90,89,1,0,0,0,91,92,1,0,
  	0,0,92,93,5,64,0,0,93,95,5,65,0,0,94,96,3,10,5,0,95,94,1,0,0,0,95,96,
  	1,0,0,0,96,101,1,0,0,0,97,98,5,69,0,0,98,100,3,10,5,0,99,97,1,0,0,0,100,
  	103,1,0,0,0,101,99,1,0,0,0,101,102,1,0,0,0,102,104,1,0,0,0,103,101,1,
  	0,0,0,104,105,5,66,0,0,105,106,5,4,0,0,106,107,5,67,0,0,107,108,3,14,
  	7,0,108,109,5,68,0,0,109,110,5,5,0,0,110,111,5,67,0,0,111,112,3,22,11,
  	0,112,113,5,68,0,0,113,5,1,0,0,0,114,115,5,6,0,0,115,116,5,65,0,0,116,
  	117,5,63,0,0,117,118,5,69,0,0,118,119,5,67,0,0,119,120,5,68,0,0,120,121,
  	5,69,0,0,121,122,5,67,0,0,122,123,3,14,7,0,123,124,5,68,0,0,124,125,5,
  	66,0,0,125,7,1,0,0,0,126,127,1,0,0,0,127,9,1,0,0,0,128,129,5,64,0,0,129,
  	130,5,4,0,0,130,131,3,14,7,0,131,11,1,0,0,0,132,136,5,67,0,0,133,135,
  	3,14,7,0,134,133,1,0,0,0,135,138,1,0,0,0,136,134,1,0,0,0,136,137,1,0,
  	0,0,137,143,1,0,0,0,138,136,1,0,0,0,139,140,5,69,0,0,140,142,3,14,7,0,
  	141,139,1,0,0,0,142,145,1,0,0,0,143,141,1,0,0,0,143,144,1,0,0,0,144,146,
  	1,0,0,0,145,143,1,0,0,0,146,147,5,68,0,0,147,13,1,0,0,0,148,149,3,20,
  	10,0,149,150,3,66,33,0,150,153,1,0,0,0,151,153,3,66,33,0,152,148,1,0,
  	0,0,152,151,1,0,0,0,153,15,1,0,0,0,154,156,5,67,0,0,155,157,3,14,7,0,
  	156,155,1,0,0,0,156,157,1,0,0,0,157,162,1,0,0,0,158,159,5,69,0,0,159,
  	161,3,14,7,0,160,158,1,0,0,0,161,164,1,0,0,0,162,160,1,0,0,0,162,163,
  	1,0,0,0,163,165,1,0,0,0,164,162,1,0,0,0,165,166,5,68,0,0,166,17,1,0,0,
  	0,167,168,3,14,7,0,168,19,1,0,0,0,169,173,5,7,0,0,170,172,3,62,31,0,171,
  	170,1,0,0,0,172,175,1,0,0,0,173,171,1,0,0,0,173,174,1,0,0,0,174,176,1,
  	0,0,0,175,173,1,0,0,0,176,178,5,8,0,0,177,169,1,0,0,0,178,179,1,0,0,0,
  	179,177,1,0,0,0,179,180,1,0,0,0,180,21,1,0,0,0,181,183,3,24,12,0,182,
  	181,1,0,0,0,183,186,1,0,0,0,184,182,1,0,0,0,184,185,1,0,0,0,185,187,1,
  	0,0,0,186,184,1,0,0,0,187,188,5,9,0,0,188,189,5,67,0,0,189,194,3,62,31,
  	0,190,191,5,69,0,0,191,193,3,62,31,0,192,190,1,0,0,0,193,196,1,0,0,0,
  	194,192,1,0,0,0,194,195,1,0,0,0,195,197,1,0,0,0,196,194,1,0,0,0,197,198,
  	5,68,0,0,198,211,1,0,0,0,199,200,5,67,0,0,200,205,3,62,31,0,201,202,5,
  	69,0,0,202,204,3,62,31,0,203,201,1,0,0,0,204,207,1,0,0,0,205,203,1,0,
  	0,0,205,206,1,0,0,0,206,208,1,0,0,0,207,205,1,0,0,0,208,209,5,68,0,0,
  	209,211,1,0,0,0,210,184,1,0,0,0,210,199,1,0,0,0,211,23,1,0,0,0,212,213,
  	5,10,0,0,213,214,3,26,13,0,214,215,5,5,0,0,215,216,3,30,15,0,216,25,1,
  	0,0,0,217,218,5,67,0,0,218,224,3,28,14,0,219,220,3,28,14,0,220,221,5,
  	69,0,0,221,223,1,0,0,0,222,219,1,0,0,0,223,226,1,0,0,0,224,222,1,0,0,
  	0,224,225,1,0,0,0,225,227,1,0,0,0,226,224,1,0,0,0,227,228,5,68,0,0,228,
  	27,1,0,0,0,229,230,5,64,0,0,230,231,5,4,0,0,231,232,3,14,7,0,232,29,1,
  	0,0,0,233,238,3,48,24,0,234,238,3,54,27,0,235,238,3,32,16,0,236,238,3,
  	62,31,0,237,233,1,0,0,0,237,234,1,0,0,0,237,235,1,0,0,0,237,236,1,0,0,
  	0,238,31,1,0,0,0,239,240,5,11,0,0,240,241,5,65,0,0,241,242,3,34,17,0,
  	242,243,3,36,18,0,243,244,5,66,0,0,244,252,1,0,0,0,245,246,5,12,0,0,246,
  	247,5,65,0,0,247,248,3,34,17,0,248,249,3,38,19,0,249,250,5,66,0,0,250,
  	252,1,0,0,0,251,239,1,0,0,0,251,245,1,0,0,0,252,33,1,0,0,0,253,254,3,
  	62,31,0,254,255,5,69,0,0,255,257,5,67,0,0,256,258,5,64,0,0,257,256,1,
  	0,0,0,257,258,1,0,0,0,258,263,1,0,0,0,259,260,5,69,0,0,260,262,5,64,0,
  	0,261,259,1,0,0,0,262,265,1,0,0,0,263,261,1,0,0,0,263,264,1,0,0,0,264,
  	266,1,0,0,0,265,263,1,0,0,0,266,267,5,68,0,0,267,268,5,69,0,0,268,35,
  	1,0,0,0,269,270,3,44,22,0,270,37,1,0,0,0,271,272,3,44,22,0,272,273,5,
  	69,0,0,273,275,5,67,0,0,274,276,3,40,20,0,275,274,1,0,0,0,275,276,1,0,
  	0,0,276,281,1,0,0,0,277,278,5,69,0,0,278,280,3,40,20,0,279,277,1,0,0,
  	0,280,283,1,0,0,0,281,279,1,0,0,0,281,282,1,0,0,0,282,284,1,0,0,0,283,
  	281,1,0,0,0,284,285,5,68,0,0,285,39,1,0,0,0,286,287,3,42,21,0,287,288,
  	3,44,22,0,288,289,5,69,0,0,289,291,5,67,0,0,290,292,3,62,31,0,291,290,
  	1,0,0,0,291,292,1,0,0,0,292,297,1,0,0,0,293,294,5,69,0,0,294,296,3,62,
  	31,0,295,293,1,0,0,0,296,299,1,0,0,0,297,295,1,0,0,0,297,298,1,0,0,0,
  	298,300,1,0,0,0,299,297,1,0,0,0,300,301,5,68,0,0,301,41,1,0,0,0,302,304,
  	5,13,0,0,303,302,1,0,0,0,303,304,1,0,0,0,304,43,1,0,0,0,305,306,5,14,
  	0,0,306,308,5,67,0,0,307,309,3,46,23,0,308,307,1,0,0,0,308,309,1,0,0,
  	0,309,314,1,0,0,0,310,311,5,69,0,0,311,313,3,46,23,0,312,310,1,0,0,0,
  	313,316,1,0,0,0,314,312,1,0,0,0,314,315,1,0,0,0,315,317,1,0,0,0,316,314,
  	1,0,0,0,317,318,5,68,0,0,318,319,5,4,0,0,319,320,3,16,8,0,320,321,5,15,
  	0,0,321,322,3,22,11,0,322,45,1,0,0,0,323,324,5,64,0,0,324,325,5,4,0,0,
  	325,326,3,14,7,0,326,47,1,0,0,0,327,328,5,16,0,0,328,329,5,64,0,0,329,
  	331,5,65,0,0,330,332,3,62,31,0,331,330,1,0,0,0,331,332,1,0,0,0,332,337,
  	1,0,0,0,333,334,5,69,0,0,334,336,3,62,31,0,335,333,1,0,0,0,336,339,1,
  	0,0,0,337,335,1,0,0,0,337,338,1,0,0,0,338,340,1,0,0,0,339,337,1,0,0,0,
  	340,341,5,66,0,0,341,342,5,4,0,0,342,343,3,12,6,0,343,49,1,0,0,0,344,
  	345,3,62,31,0,345,51,1,0,0,0,346,347,5,7,0,0,347,352,3,62,31,0,348,349,
  	5,69,0,0,349,351,3,62,31,0,350,348,1,0,0,0,351,354,1,0,0,0,352,350,1,
  	0,0,0,352,353,1,0,0,0,353,355,1,0,0,0,354,352,1,0,0,0,355,356,5,8,0,0,
  	356,53,1,0,0,0,357,358,5,17,0,0,358,359,5,65,0,0,359,360,3,20,10,0,360,
  	361,5,69,0,0,361,362,3,62,31,0,362,363,5,66,0,0,363,433,1,0,0,0,364,365,
  	3,52,26,0,365,366,5,4,0,0,366,367,3,14,7,0,367,433,1,0,0,0,368,369,3,
  	62,31,0,369,370,5,7,0,0,370,375,3,62,31,0,371,372,5,69,0,0,372,374,3,
  	62,31,0,373,371,1,0,0,0,374,377,1,0,0,0,375,373,1,0,0,0,375,376,1,0,0,
  	0,376,378,1,0,0,0,377,375,1,0,0,0,378,379,5,8,0,0,379,433,1,0,0,0,380,
  	381,5,29,0,0,381,382,5,65,0,0,382,383,3,62,31,0,383,384,5,69,0,0,384,
  	385,3,62,31,0,385,386,5,69,0,0,386,387,3,62,31,0,387,388,5,66,0,0,388,
  	433,1,0,0,0,389,390,5,28,0,0,390,391,5,65,0,0,391,392,3,62,31,0,392,393,
  	5,69,0,0,393,394,3,62,31,0,394,395,5,69,0,0,395,396,3,62,31,0,396,397,
  	5,66,0,0,397,433,1,0,0,0,398,433,3,56,28,0,399,433,3,60,30,0,400,401,
  	5,18,0,0,401,402,5,65,0,0,402,403,3,62,31,0,403,404,5,69,0,0,404,405,
  	5,65,0,0,405,406,5,27,0,0,406,407,5,19,0,0,407,408,5,27,0,0,408,409,5,
  	20,0,0,409,410,3,20,10,0,410,411,5,21,0,0,411,412,1,0,0,0,412,413,3,20,
  	10,0,413,414,5,66,0,0,414,415,5,66,0,0,415,433,1,0,0,0,416,417,5,22,0,
  	0,417,418,5,65,0,0,418,419,3,62,31,0,419,420,5,69,0,0,420,421,5,65,0,
  	0,421,426,5,27,0,0,422,423,5,69,0,0,423,425,5,27,0,0,424,422,1,0,0,0,
  	425,428,1,0,0,0,426,424,1,0,0,0,426,427,1,0,0,0,427,429,1,0,0,0,428,426,
  	1,0,0,0,429,430,5,66,0,0,430,431,5,66,0,0,431,433,1,0,0,0,432,357,1,0,
  	0,0,432,364,1,0,0,0,432,368,1,0,0,0,432,380,1,0,0,0,432,389,1,0,0,0,432,
  	398,1,0,0,0,432,399,1,0,0,0,432,400,1,0,0,0,432,416,1,0,0,0,433,55,1,
  	0,0,0,434,435,3,58,29,0,435,436,5,65,0,0,436,437,3,62,31,0,437,438,5,
  	69,0,0,438,439,3,62,31,0,439,440,5,66,0,0,440,57,1,0,0,0,441,442,7,0,
  	0,0,442,59,1,0,0,0,443,444,5,23,0,0,444,445,3,66,33,0,445,446,3,62,31,
  	0,446,447,5,24,0,0,447,448,3,66,33,0,448,61,1,0,0,0,449,452,3,64,32,0,
  	450,452,5,64,0,0,451,449,1,0,0,0,451,450,1,0,0,0,452,63,1,0,0,0,453,454,
  	5,27,0,0,454,458,5,25,0,0,455,456,5,62,0,0,456,458,5,26,0,0,457,453,1,
  	0,0,0,457,455,1,0,0,0,458,65,1,0,0,0,459,462,5,25,0,0,460,462,5,26,0,
  	0,461,459,1,0,0,0,461,460,1,0,0,0,462,67,1,0,0,0,38,69,74,87,90,95,101,
  	136,143,152,156,162,173,179,184,194,205,210,224,237,251,257,263,275,281,
  	291,297,303,308,314,331,337,352,375,426,432,451,457,461
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  futharkParserStaticData = std::move(staticData);
}

}

FutharkParser::FutharkParser(TokenStream *input) : FutharkParser(input, antlr4::atn::ParserATNSimulatorOptions()) {}

FutharkParser::FutharkParser(TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options) : Parser(input) {
  FutharkParser::initialize();
  _interpreter = new atn::ParserATNSimulator(this, *futharkParserStaticData->atn, futharkParserStaticData->decisionToDFA, futharkParserStaticData->sharedContextCache, options);
}

FutharkParser::~FutharkParser() {
  delete _interpreter;
}

const atn::ATN& FutharkParser::getATN() const {
  return *futharkParserStaticData->atn;
}

std::string FutharkParser::getGrammarFileName() const {
  return "Futhark.g4";
}

const std::vector<std::string>& FutharkParser::getRuleNames() const {
  return futharkParserStaticData->ruleNames;
}

const dfa::Vocabulary& FutharkParser::getVocabulary() const {
  return futharkParserStaticData->vocabulary;
}

antlr4::atn::SerializedATNView FutharkParser::getSerializedATN() const {
  return futharkParserStaticData->serializedATN;
}


//----------------- RootContext ------------------------------------------------------------------

FutharkParser::RootContext::RootContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FutharkParser::RootContext::EOF() {
  return getToken(FutharkParser::EOF, 0);
}

FutharkParser::HeaderContext* FutharkParser::RootContext::header() {
  return getRuleContext<FutharkParser::HeaderContext>(0);
}

std::vector<FutharkParser::PFunDefContext *> FutharkParser::RootContext::pFunDef() {
  return getRuleContexts<FutharkParser::PFunDefContext>();
}

FutharkParser::PFunDefContext* FutharkParser::RootContext::pFunDef(size_t i) {
  return getRuleContext<FutharkParser::PFunDefContext>(i);
}


size_t FutharkParser::RootContext::getRuleIndex() const {
  return FutharkParser::RuleRoot;
}

void FutharkParser::RootContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRoot(this);
}

void FutharkParser::RootContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRoot(this);
}

FutharkParser::RootContext* FutharkParser::root() {
  RootContext *_localctx = _tracker.createInstance<RootContext>(_ctx, getState());
  enterRule(_localctx, 0, FutharkParser::RuleRoot);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(69);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == FutharkParser::T__0) {
      setState(68);
      header();
    }
    setState(72); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(71);
      pFunDef();
      setState(74); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (((((_la - 3) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 3)) & 2305843009213693961) != 0));
    setState(76);
    match(FutharkParser::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- HeaderContext ------------------------------------------------------------------

FutharkParser::HeaderContext::HeaderContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> FutharkParser::HeaderContext::OPEN_BRACKET() {
  return getTokens(FutharkParser::OPEN_BRACKET);
}

tree::TerminalNode* FutharkParser::HeaderContext::OPEN_BRACKET(size_t i) {
  return getToken(FutharkParser::OPEN_BRACKET, i);
}

tree::TerminalNode* FutharkParser::HeaderContext::NUMBER() {
  return getToken(FutharkParser::NUMBER, 0);
}

std::vector<tree::TerminalNode *> FutharkParser::HeaderContext::CLOSE_BRACKET() {
  return getTokens(FutharkParser::CLOSE_BRACKET);
}

tree::TerminalNode* FutharkParser::HeaderContext::CLOSE_BRACKET(size_t i) {
  return getToken(FutharkParser::CLOSE_BRACKET, i);
}


size_t FutharkParser::HeaderContext::getRuleIndex() const {
  return FutharkParser::RuleHeader;
}

void FutharkParser::HeaderContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterHeader(this);
}

void FutharkParser::HeaderContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitHeader(this);
}

FutharkParser::HeaderContext* FutharkParser::header() {
  HeaderContext *_localctx = _tracker.createInstance<HeaderContext>(_ctx, getState());
  enterRule(_localctx, 2, FutharkParser::RuleHeader);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(78);
    match(FutharkParser::T__0);
    setState(79);
    match(FutharkParser::OPEN_BRACKET);
    setState(80);
    match(FutharkParser::NUMBER);
    setState(81);
    match(FutharkParser::CLOSE_BRACKET);
    setState(82);
    match(FutharkParser::T__1);
    setState(83);
    match(FutharkParser::OPEN_BRACKET);
    setState(84);
    match(FutharkParser::CLOSE_BRACKET);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PFunDefContext ------------------------------------------------------------------

FutharkParser::PFunDefContext::PFunDefContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FutharkParser::PFunDefContext::ID() {
  return getToken(FutharkParser::ID, 0);
}

tree::TerminalNode* FutharkParser::PFunDefContext::LPARAM() {
  return getToken(FutharkParser::LPARAM, 0);
}

tree::TerminalNode* FutharkParser::PFunDefContext::RPARAM() {
  return getToken(FutharkParser::RPARAM, 0);
}

std::vector<tree::TerminalNode *> FutharkParser::PFunDefContext::OPEN_BRACKET() {
  return getTokens(FutharkParser::OPEN_BRACKET);
}

tree::TerminalNode* FutharkParser::PFunDefContext::OPEN_BRACKET(size_t i) {
  return getToken(FutharkParser::OPEN_BRACKET, i);
}

FutharkParser::PTypeContext* FutharkParser::PFunDefContext::pType() {
  return getRuleContext<FutharkParser::PTypeContext>(0);
}

std::vector<tree::TerminalNode *> FutharkParser::PFunDefContext::CLOSE_BRACKET() {
  return getTokens(FutharkParser::CLOSE_BRACKET);
}

tree::TerminalNode* FutharkParser::PFunDefContext::CLOSE_BRACKET(size_t i) {
  return getToken(FutharkParser::CLOSE_BRACKET, i);
}

FutharkParser::PBodyContext* FutharkParser::PFunDefContext::pBody() {
  return getRuleContext<FutharkParser::PBodyContext>(0);
}

std::vector<FutharkParser::FParamContext *> FutharkParser::PFunDefContext::fParam() {
  return getRuleContexts<FutharkParser::FParamContext>();
}

FutharkParser::FParamContext* FutharkParser::PFunDefContext::fParam(size_t i) {
  return getRuleContext<FutharkParser::FParamContext>(i);
}

std::vector<tree::TerminalNode *> FutharkParser::PFunDefContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::PFunDefContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}

FutharkParser::PEntryContext* FutharkParser::PFunDefContext::pEntry() {
  return getRuleContext<FutharkParser::PEntryContext>(0);
}


size_t FutharkParser::PFunDefContext::getRuleIndex() const {
  return FutharkParser::RulePFunDef;
}

void FutharkParser::PFunDefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPFunDef(this);
}

void FutharkParser::PFunDefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPFunDef(this);
}

FutharkParser::PFunDefContext* FutharkParser::pFunDef() {
  PFunDefContext *_localctx = _tracker.createInstance<PFunDefContext>(_ctx, getState());
  enterRule(_localctx, 4, FutharkParser::RulePFunDef);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(90);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case FutharkParser::T__5:
      case FutharkParser::ID: {
        setState(87);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == FutharkParser::T__5) {
          setState(86);
          pEntry();
        }
        break;
      }

      case FutharkParser::T__2: {
        setState(89);
        match(FutharkParser::T__2);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    setState(92);
    match(FutharkParser::ID);
    setState(93);
    match(FutharkParser::LPARAM);
    setState(95);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == FutharkParser::ID) {
      setState(94);
      fParam();
    }
    setState(101);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == FutharkParser::COMMA) {
      setState(97);
      match(FutharkParser::COMMA);
      setState(98);
      fParam();
      setState(103);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(104);
    match(FutharkParser::RPARAM);
    setState(105);
    match(FutharkParser::T__3);
    setState(106);
    match(FutharkParser::OPEN_BRACKET);
    setState(107);
    pType();
    setState(108);
    match(FutharkParser::CLOSE_BRACKET);
    setState(109);
    match(FutharkParser::T__4);
    setState(110);
    match(FutharkParser::OPEN_BRACKET);
    setState(111);
    pBody();
    setState(112);
    match(FutharkParser::CLOSE_BRACKET);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PEntryContext ------------------------------------------------------------------

FutharkParser::PEntryContext::PEntryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FutharkParser::PEntryContext::LPARAM() {
  return getToken(FutharkParser::LPARAM, 0);
}

tree::TerminalNode* FutharkParser::PEntryContext::STRING_LITERAL() {
  return getToken(FutharkParser::STRING_LITERAL, 0);
}

std::vector<tree::TerminalNode *> FutharkParser::PEntryContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::PEntryContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}

std::vector<tree::TerminalNode *> FutharkParser::PEntryContext::OPEN_BRACKET() {
  return getTokens(FutharkParser::OPEN_BRACKET);
}

tree::TerminalNode* FutharkParser::PEntryContext::OPEN_BRACKET(size_t i) {
  return getToken(FutharkParser::OPEN_BRACKET, i);
}

std::vector<tree::TerminalNode *> FutharkParser::PEntryContext::CLOSE_BRACKET() {
  return getTokens(FutharkParser::CLOSE_BRACKET);
}

tree::TerminalNode* FutharkParser::PEntryContext::CLOSE_BRACKET(size_t i) {
  return getToken(FutharkParser::CLOSE_BRACKET, i);
}

FutharkParser::PTypeContext* FutharkParser::PEntryContext::pType() {
  return getRuleContext<FutharkParser::PTypeContext>(0);
}

tree::TerminalNode* FutharkParser::PEntryContext::RPARAM() {
  return getToken(FutharkParser::RPARAM, 0);
}


size_t FutharkParser::PEntryContext::getRuleIndex() const {
  return FutharkParser::RulePEntry;
}

void FutharkParser::PEntryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPEntry(this);
}

void FutharkParser::PEntryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPEntry(this);
}

FutharkParser::PEntryContext* FutharkParser::pEntry() {
  PEntryContext *_localctx = _tracker.createInstance<PEntryContext>(_ctx, getState());
  enterRule(_localctx, 6, FutharkParser::RulePEntry);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(114);
    match(FutharkParser::T__5);
    setState(115);
    match(FutharkParser::LPARAM);
    setState(116);
    match(FutharkParser::STRING_LITERAL);
    setState(117);
    match(FutharkParser::COMMA);
    setState(118);
    match(FutharkParser::OPEN_BRACKET);
    setState(119);
    match(FutharkParser::CLOSE_BRACKET);
    setState(120);
    match(FutharkParser::COMMA);
    setState(121);
    match(FutharkParser::OPEN_BRACKET);
    setState(122);
    pType();
    setState(123);
    match(FutharkParser::CLOSE_BRACKET);
    setState(124);
    match(FutharkParser::RPARAM);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PEntryPointInputContext ------------------------------------------------------------------

FutharkParser::PEntryPointInputContext::PEntryPointInputContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t FutharkParser::PEntryPointInputContext::getRuleIndex() const {
  return FutharkParser::RulePEntryPointInput;
}

void FutharkParser::PEntryPointInputContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPEntryPointInput(this);
}

void FutharkParser::PEntryPointInputContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPEntryPointInput(this);
}

FutharkParser::PEntryPointInputContext* FutharkParser::pEntryPointInput() {
  PEntryPointInputContext *_localctx = _tracker.createInstance<PEntryPointInputContext>(_ctx, getState());
  enterRule(_localctx, 8, FutharkParser::RulePEntryPointInput);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);

   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FParamContext ------------------------------------------------------------------

FutharkParser::FParamContext::FParamContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FutharkParser::FParamContext::ID() {
  return getToken(FutharkParser::ID, 0);
}

FutharkParser::PTypeContext* FutharkParser::FParamContext::pType() {
  return getRuleContext<FutharkParser::PTypeContext>(0);
}


size_t FutharkParser::FParamContext::getRuleIndex() const {
  return FutharkParser::RuleFParam;
}

void FutharkParser::FParamContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFParam(this);
}

void FutharkParser::FParamContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFParam(this);
}

FutharkParser::FParamContext* FutharkParser::fParam() {
  FParamContext *_localctx = _tracker.createInstance<FParamContext>(_ctx, getState());
  enterRule(_localctx, 10, FutharkParser::RuleFParam);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(128);
    match(FutharkParser::ID);
    setState(129);
    match(FutharkParser::T__3);
    setState(130);
    pType();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PRetTypesContext ------------------------------------------------------------------

FutharkParser::PRetTypesContext::PRetTypesContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FutharkParser::PRetTypesContext::OPEN_BRACKET() {
  return getToken(FutharkParser::OPEN_BRACKET, 0);
}

tree::TerminalNode* FutharkParser::PRetTypesContext::CLOSE_BRACKET() {
  return getToken(FutharkParser::CLOSE_BRACKET, 0);
}

std::vector<FutharkParser::PTypeContext *> FutharkParser::PRetTypesContext::pType() {
  return getRuleContexts<FutharkParser::PTypeContext>();
}

FutharkParser::PTypeContext* FutharkParser::PRetTypesContext::pType(size_t i) {
  return getRuleContext<FutharkParser::PTypeContext>(i);
}

std::vector<tree::TerminalNode *> FutharkParser::PRetTypesContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::PRetTypesContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}


size_t FutharkParser::PRetTypesContext::getRuleIndex() const {
  return FutharkParser::RulePRetTypes;
}

void FutharkParser::PRetTypesContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPRetTypes(this);
}

void FutharkParser::PRetTypesContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPRetTypes(this);
}

FutharkParser::PRetTypesContext* FutharkParser::pRetTypes() {
  PRetTypesContext *_localctx = _tracker.createInstance<PRetTypesContext>(_ctx, getState());
  enterRule(_localctx, 12, FutharkParser::RulePRetTypes);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(132);
    match(FutharkParser::OPEN_BRACKET);
    setState(136);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 100663424) != 0)) {
      setState(133);
      pType();
      setState(138);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(143);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == FutharkParser::COMMA) {
      setState(139);
      match(FutharkParser::COMMA);
      setState(140);
      pType();
      setState(145);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(146);
    match(FutharkParser::CLOSE_BRACKET);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PTypeContext ------------------------------------------------------------------

FutharkParser::PTypeContext::PTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t FutharkParser::PTypeContext::getRuleIndex() const {
  return FutharkParser::RulePType;
}

void FutharkParser::PTypeContext::copyFrom(PTypeContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- TypePrimContext ------------------------------------------------------------------

FutharkParser::PPrimTypeContext* FutharkParser::TypePrimContext::pPrimType() {
  return getRuleContext<FutharkParser::PPrimTypeContext>(0);
}

FutharkParser::TypePrimContext::TypePrimContext(PTypeContext *ctx) { copyFrom(ctx); }

void FutharkParser::TypePrimContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypePrim(this);
}
void FutharkParser::TypePrimContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypePrim(this);
}
//----------------- TypeShapeContext ------------------------------------------------------------------

FutharkParser::PExtShapeContext* FutharkParser::TypeShapeContext::pExtShape() {
  return getRuleContext<FutharkParser::PExtShapeContext>(0);
}

FutharkParser::PPrimTypeContext* FutharkParser::TypeShapeContext::pPrimType() {
  return getRuleContext<FutharkParser::PPrimTypeContext>(0);
}

FutharkParser::TypeShapeContext::TypeShapeContext(PTypeContext *ctx) { copyFrom(ctx); }

void FutharkParser::TypeShapeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypeShape(this);
}
void FutharkParser::TypeShapeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypeShape(this);
}
FutharkParser::PTypeContext* FutharkParser::pType() {
  PTypeContext *_localctx = _tracker.createInstance<PTypeContext>(_ctx, getState());
  enterRule(_localctx, 14, FutharkParser::RulePType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(152);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case FutharkParser::T__6: {
        _localctx = _tracker.createInstance<FutharkParser::TypeShapeContext>(_localctx);
        enterOuterAlt(_localctx, 1);
        setState(148);
        pExtShape();
        setState(149);
        pPrimType();
        break;
      }

      case FutharkParser::INTEGER_TYPE:
      case FutharkParser::FLOAT_TYPE: {
        _localctx = _tracker.createInstance<FutharkParser::TypePrimContext>(_localctx);
        enterOuterAlt(_localctx, 2);
        setState(151);
        pPrimType();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PTypesContext ------------------------------------------------------------------

FutharkParser::PTypesContext::PTypesContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FutharkParser::PTypesContext::OPEN_BRACKET() {
  return getToken(FutharkParser::OPEN_BRACKET, 0);
}

tree::TerminalNode* FutharkParser::PTypesContext::CLOSE_BRACKET() {
  return getToken(FutharkParser::CLOSE_BRACKET, 0);
}

std::vector<FutharkParser::PTypeContext *> FutharkParser::PTypesContext::pType() {
  return getRuleContexts<FutharkParser::PTypeContext>();
}

FutharkParser::PTypeContext* FutharkParser::PTypesContext::pType(size_t i) {
  return getRuleContext<FutharkParser::PTypeContext>(i);
}

std::vector<tree::TerminalNode *> FutharkParser::PTypesContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::PTypesContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}


size_t FutharkParser::PTypesContext::getRuleIndex() const {
  return FutharkParser::RulePTypes;
}

void FutharkParser::PTypesContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPTypes(this);
}

void FutharkParser::PTypesContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPTypes(this);
}

FutharkParser::PTypesContext* FutharkParser::pTypes() {
  PTypesContext *_localctx = _tracker.createInstance<PTypesContext>(_ctx, getState());
  enterRule(_localctx, 16, FutharkParser::RulePTypes);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(154);
    match(FutharkParser::OPEN_BRACKET);
    setState(156);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 100663424) != 0)) {
      setState(155);
      pType();
    }
    setState(162);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == FutharkParser::COMMA) {
      setState(158);
      match(FutharkParser::COMMA);
      setState(159);
      pType();
      setState(164);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(165);
    match(FutharkParser::CLOSE_BRACKET);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PRetTypeContext ------------------------------------------------------------------

FutharkParser::PRetTypeContext::PRetTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

FutharkParser::PTypeContext* FutharkParser::PRetTypeContext::pType() {
  return getRuleContext<FutharkParser::PTypeContext>(0);
}


size_t FutharkParser::PRetTypeContext::getRuleIndex() const {
  return FutharkParser::RulePRetType;
}

void FutharkParser::PRetTypeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPRetType(this);
}

void FutharkParser::PRetTypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPRetType(this);
}

FutharkParser::PRetTypeContext* FutharkParser::pRetType() {
  PRetTypeContext *_localctx = _tracker.createInstance<PRetTypeContext>(_ctx, getState());
  enterRule(_localctx, 18, FutharkParser::RulePRetType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(167);
    pType();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PExtShapeContext ------------------------------------------------------------------

FutharkParser::PExtShapeContext::PExtShapeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<FutharkParser::PSubExpContext *> FutharkParser::PExtShapeContext::pSubExp() {
  return getRuleContexts<FutharkParser::PSubExpContext>();
}

FutharkParser::PSubExpContext* FutharkParser::PExtShapeContext::pSubExp(size_t i) {
  return getRuleContext<FutharkParser::PSubExpContext>(i);
}


size_t FutharkParser::PExtShapeContext::getRuleIndex() const {
  return FutharkParser::RulePExtShape;
}

void FutharkParser::PExtShapeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPExtShape(this);
}

void FutharkParser::PExtShapeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPExtShape(this);
}

FutharkParser::PExtShapeContext* FutharkParser::pExtShape() {
  PExtShapeContext *_localctx = _tracker.createInstance<PExtShapeContext>(_ctx, getState());
  enterRule(_localctx, 20, FutharkParser::RulePExtShape);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(177); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(169);
      match(FutharkParser::T__6);
      setState(173);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (((((_la - 27) & ~ 0x3fULL) == 0) &&
        ((1ULL << (_la - 27)) & 171798691841) != 0)) {
        setState(170);
        pSubExp();
        setState(175);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(176);
      match(FutharkParser::T__7);
      setState(179); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == FutharkParser::T__6);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PBodyContext ------------------------------------------------------------------

FutharkParser::PBodyContext::PBodyContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t FutharkParser::PBodyContext::getRuleIndex() const {
  return FutharkParser::RulePBody;
}

void FutharkParser::PBodyContext::copyFrom(PBodyContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- EmptyBodyContext ------------------------------------------------------------------

tree::TerminalNode* FutharkParser::EmptyBodyContext::OPEN_BRACKET() {
  return getToken(FutharkParser::OPEN_BRACKET, 0);
}

std::vector<FutharkParser::PSubExpContext *> FutharkParser::EmptyBodyContext::pSubExp() {
  return getRuleContexts<FutharkParser::PSubExpContext>();
}

FutharkParser::PSubExpContext* FutharkParser::EmptyBodyContext::pSubExp(size_t i) {
  return getRuleContext<FutharkParser::PSubExpContext>(i);
}

tree::TerminalNode* FutharkParser::EmptyBodyContext::CLOSE_BRACKET() {
  return getToken(FutharkParser::CLOSE_BRACKET, 0);
}

std::vector<tree::TerminalNode *> FutharkParser::EmptyBodyContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::EmptyBodyContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}

FutharkParser::EmptyBodyContext::EmptyBodyContext(PBodyContext *ctx) { copyFrom(ctx); }

void FutharkParser::EmptyBodyContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEmptyBody(this);
}
void FutharkParser::EmptyBodyContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEmptyBody(this);
}
//----------------- BodyContext ------------------------------------------------------------------

tree::TerminalNode* FutharkParser::BodyContext::OPEN_BRACKET() {
  return getToken(FutharkParser::OPEN_BRACKET, 0);
}

std::vector<FutharkParser::PSubExpContext *> FutharkParser::BodyContext::pSubExp() {
  return getRuleContexts<FutharkParser::PSubExpContext>();
}

FutharkParser::PSubExpContext* FutharkParser::BodyContext::pSubExp(size_t i) {
  return getRuleContext<FutharkParser::PSubExpContext>(i);
}

tree::TerminalNode* FutharkParser::BodyContext::CLOSE_BRACKET() {
  return getToken(FutharkParser::CLOSE_BRACKET, 0);
}

std::vector<FutharkParser::PStmContext *> FutharkParser::BodyContext::pStm() {
  return getRuleContexts<FutharkParser::PStmContext>();
}

FutharkParser::PStmContext* FutharkParser::BodyContext::pStm(size_t i) {
  return getRuleContext<FutharkParser::PStmContext>(i);
}

std::vector<tree::TerminalNode *> FutharkParser::BodyContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::BodyContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}

FutharkParser::BodyContext::BodyContext(PBodyContext *ctx) { copyFrom(ctx); }

void FutharkParser::BodyContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBody(this);
}
void FutharkParser::BodyContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBody(this);
}
FutharkParser::PBodyContext* FutharkParser::pBody() {
  PBodyContext *_localctx = _tracker.createInstance<PBodyContext>(_ctx, getState());
  enterRule(_localctx, 22, FutharkParser::RulePBody);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(210);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case FutharkParser::T__8:
      case FutharkParser::T__9: {
        _localctx = _tracker.createInstance<FutharkParser::BodyContext>(_localctx);
        enterOuterAlt(_localctx, 1);
        setState(184);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == FutharkParser::T__9) {
          setState(181);
          pStm();
          setState(186);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(187);
        match(FutharkParser::T__8);
        setState(188);
        match(FutharkParser::OPEN_BRACKET);
        setState(189);
        pSubExp();
        setState(194);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == FutharkParser::COMMA) {
          setState(190);
          match(FutharkParser::COMMA);
          setState(191);
          pSubExp();
          setState(196);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(197);
        match(FutharkParser::CLOSE_BRACKET);
        break;
      }

      case FutharkParser::OPEN_BRACKET: {
        _localctx = _tracker.createInstance<FutharkParser::EmptyBodyContext>(_localctx);
        enterOuterAlt(_localctx, 2);
        setState(199);
        match(FutharkParser::OPEN_BRACKET);
        setState(200);
        pSubExp();
        setState(205);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == FutharkParser::COMMA) {
          setState(201);
          match(FutharkParser::COMMA);
          setState(202);
          pSubExp();
          setState(207);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(208);
        match(FutharkParser::CLOSE_BRACKET);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PStmContext ------------------------------------------------------------------

FutharkParser::PStmContext::PStmContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

FutharkParser::PPatContext* FutharkParser::PStmContext::pPat() {
  return getRuleContext<FutharkParser::PPatContext>(0);
}

FutharkParser::PExpContext* FutharkParser::PStmContext::pExp() {
  return getRuleContext<FutharkParser::PExpContext>(0);
}


size_t FutharkParser::PStmContext::getRuleIndex() const {
  return FutharkParser::RulePStm;
}

void FutharkParser::PStmContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPStm(this);
}

void FutharkParser::PStmContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPStm(this);
}

FutharkParser::PStmContext* FutharkParser::pStm() {
  PStmContext *_localctx = _tracker.createInstance<PStmContext>(_ctx, getState());
  enterRule(_localctx, 24, FutharkParser::RulePStm);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(212);
    match(FutharkParser::T__9);
    setState(213);
    pPat();
    setState(214);
    match(FutharkParser::T__4);
    setState(215);
    pExp();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PPatContext ------------------------------------------------------------------

FutharkParser::PPatContext::PPatContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FutharkParser::PPatContext::OPEN_BRACKET() {
  return getToken(FutharkParser::OPEN_BRACKET, 0);
}

std::vector<FutharkParser::PPatElemContext *> FutharkParser::PPatContext::pPatElem() {
  return getRuleContexts<FutharkParser::PPatElemContext>();
}

FutharkParser::PPatElemContext* FutharkParser::PPatContext::pPatElem(size_t i) {
  return getRuleContext<FutharkParser::PPatElemContext>(i);
}

tree::TerminalNode* FutharkParser::PPatContext::CLOSE_BRACKET() {
  return getToken(FutharkParser::CLOSE_BRACKET, 0);
}

std::vector<tree::TerminalNode *> FutharkParser::PPatContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::PPatContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}


size_t FutharkParser::PPatContext::getRuleIndex() const {
  return FutharkParser::RulePPat;
}

void FutharkParser::PPatContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPPat(this);
}

void FutharkParser::PPatContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPPat(this);
}

FutharkParser::PPatContext* FutharkParser::pPat() {
  PPatContext *_localctx = _tracker.createInstance<PPatContext>(_ctx, getState());
  enterRule(_localctx, 26, FutharkParser::RulePPat);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(217);
    match(FutharkParser::OPEN_BRACKET);
    setState(218);
    pPatElem();
    setState(224);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == FutharkParser::ID) {
      setState(219);
      pPatElem();
      setState(220);
      match(FutharkParser::COMMA);
      setState(226);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(227);
    match(FutharkParser::CLOSE_BRACKET);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PPatElemContext ------------------------------------------------------------------

FutharkParser::PPatElemContext::PPatElemContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FutharkParser::PPatElemContext::ID() {
  return getToken(FutharkParser::ID, 0);
}

FutharkParser::PTypeContext* FutharkParser::PPatElemContext::pType() {
  return getRuleContext<FutharkParser::PTypeContext>(0);
}


size_t FutharkParser::PPatElemContext::getRuleIndex() const {
  return FutharkParser::RulePPatElem;
}

void FutharkParser::PPatElemContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPPatElem(this);
}

void FutharkParser::PPatElemContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPPatElem(this);
}

FutharkParser::PPatElemContext* FutharkParser::pPatElem() {
  PPatElemContext *_localctx = _tracker.createInstance<PPatElemContext>(_ctx, getState());
  enterRule(_localctx, 28, FutharkParser::RulePPatElem);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(229);
    match(FutharkParser::ID);
    setState(230);
    match(FutharkParser::T__3);
    setState(231);
    pType();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PExpContext ------------------------------------------------------------------

FutharkParser::PExpContext::PExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t FutharkParser::PExpContext::getRuleIndex() const {
  return FutharkParser::RulePExp;
}

void FutharkParser::PExpContext::copyFrom(PExpContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- ExpSubExpContext ------------------------------------------------------------------

FutharkParser::PSubExpContext* FutharkParser::ExpSubExpContext::pSubExp() {
  return getRuleContext<FutharkParser::PSubExpContext>(0);
}

FutharkParser::ExpSubExpContext::ExpSubExpContext(PExpContext *ctx) { copyFrom(ctx); }

void FutharkParser::ExpSubExpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpSubExp(this);
}
void FutharkParser::ExpSubExpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpSubExp(this);
}
//----------------- ExpApplyContext ------------------------------------------------------------------

FutharkParser::PApplyContext* FutharkParser::ExpApplyContext::pApply() {
  return getRuleContext<FutharkParser::PApplyContext>(0);
}

FutharkParser::ExpApplyContext::ExpApplyContext(PExpContext *ctx) { copyFrom(ctx); }

void FutharkParser::ExpApplyContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpApply(this);
}
void FutharkParser::ExpApplyContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpApply(this);
}
//----------------- ExpBasicOpContext ------------------------------------------------------------------

FutharkParser::PBasicOpContext* FutharkParser::ExpBasicOpContext::pBasicOp() {
  return getRuleContext<FutharkParser::PBasicOpContext>(0);
}

FutharkParser::ExpBasicOpContext::ExpBasicOpContext(PExpContext *ctx) { copyFrom(ctx); }

void FutharkParser::ExpBasicOpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpBasicOp(this);
}
void FutharkParser::ExpBasicOpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpBasicOp(this);
}
//----------------- ExpSoacOpContext ------------------------------------------------------------------

FutharkParser::PSoacOpContext* FutharkParser::ExpSoacOpContext::pSoacOp() {
  return getRuleContext<FutharkParser::PSoacOpContext>(0);
}

FutharkParser::ExpSoacOpContext::ExpSoacOpContext(PExpContext *ctx) { copyFrom(ctx); }

void FutharkParser::ExpSoacOpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpSoacOp(this);
}
void FutharkParser::ExpSoacOpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpSoacOp(this);
}
FutharkParser::PExpContext* FutharkParser::pExp() {
  PExpContext *_localctx = _tracker.createInstance<PExpContext>(_ctx, getState());
  enterRule(_localctx, 30, FutharkParser::RulePExp);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(237);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 18, _ctx)) {
    case 1: {
      _localctx = _tracker.createInstance<FutharkParser::ExpApplyContext>(_localctx);
      enterOuterAlt(_localctx, 1);
      setState(233);
      pApply();
      break;
    }

    case 2: {
      _localctx = _tracker.createInstance<FutharkParser::ExpBasicOpContext>(_localctx);
      enterOuterAlt(_localctx, 2);
      setState(234);
      pBasicOp();
      break;
    }

    case 3: {
      _localctx = _tracker.createInstance<FutharkParser::ExpSoacOpContext>(_localctx);
      enterOuterAlt(_localctx, 3);
      setState(235);
      pSoacOp();
      break;
    }

    case 4: {
      _localctx = _tracker.createInstance<FutharkParser::ExpSubExpContext>(_localctx);
      enterOuterAlt(_localctx, 4);
      setState(236);
      pSubExp();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PSoacOpContext ------------------------------------------------------------------

FutharkParser::PSoacOpContext::PSoacOpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t FutharkParser::PSoacOpContext::getRuleIndex() const {
  return FutharkParser::RulePSoacOp;
}

void FutharkParser::PSoacOpContext::copyFrom(PSoacOpContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- SoacOpRedomapContext ------------------------------------------------------------------

tree::TerminalNode* FutharkParser::SoacOpRedomapContext::LPARAM() {
  return getToken(FutharkParser::LPARAM, 0);
}

FutharkParser::PScremaContext* FutharkParser::SoacOpRedomapContext::pScrema() {
  return getRuleContext<FutharkParser::PScremaContext>(0);
}

FutharkParser::PRedomapFormContext* FutharkParser::SoacOpRedomapContext::pRedomapForm() {
  return getRuleContext<FutharkParser::PRedomapFormContext>(0);
}

tree::TerminalNode* FutharkParser::SoacOpRedomapContext::RPARAM() {
  return getToken(FutharkParser::RPARAM, 0);
}

FutharkParser::SoacOpRedomapContext::SoacOpRedomapContext(PSoacOpContext *ctx) { copyFrom(ctx); }

void FutharkParser::SoacOpRedomapContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSoacOpRedomap(this);
}
void FutharkParser::SoacOpRedomapContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSoacOpRedomap(this);
}
//----------------- SoacOpMapContext ------------------------------------------------------------------

tree::TerminalNode* FutharkParser::SoacOpMapContext::LPARAM() {
  return getToken(FutharkParser::LPARAM, 0);
}

FutharkParser::PScremaContext* FutharkParser::SoacOpMapContext::pScrema() {
  return getRuleContext<FutharkParser::PScremaContext>(0);
}

FutharkParser::PMapFormContext* FutharkParser::SoacOpMapContext::pMapForm() {
  return getRuleContext<FutharkParser::PMapFormContext>(0);
}

tree::TerminalNode* FutharkParser::SoacOpMapContext::RPARAM() {
  return getToken(FutharkParser::RPARAM, 0);
}

FutharkParser::SoacOpMapContext::SoacOpMapContext(PSoacOpContext *ctx) { copyFrom(ctx); }

void FutharkParser::SoacOpMapContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSoacOpMap(this);
}
void FutharkParser::SoacOpMapContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSoacOpMap(this);
}
FutharkParser::PSoacOpContext* FutharkParser::pSoacOp() {
  PSoacOpContext *_localctx = _tracker.createInstance<PSoacOpContext>(_ctx, getState());
  enterRule(_localctx, 32, FutharkParser::RulePSoacOp);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(251);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case FutharkParser::T__10: {
        _localctx = _tracker.createInstance<FutharkParser::SoacOpMapContext>(_localctx);
        enterOuterAlt(_localctx, 1);
        setState(239);
        match(FutharkParser::T__10);
        setState(240);
        match(FutharkParser::LPARAM);
        setState(241);
        pScrema();
        setState(242);
        pMapForm();
        setState(243);
        match(FutharkParser::RPARAM);
        break;
      }

      case FutharkParser::T__11: {
        _localctx = _tracker.createInstance<FutharkParser::SoacOpRedomapContext>(_localctx);
        enterOuterAlt(_localctx, 2);
        setState(245);
        match(FutharkParser::T__11);
        setState(246);
        match(FutharkParser::LPARAM);
        setState(247);
        pScrema();
        setState(248);
        pRedomapForm();
        setState(249);
        match(FutharkParser::RPARAM);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PScremaContext ------------------------------------------------------------------

FutharkParser::PScremaContext::PScremaContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

FutharkParser::PSubExpContext* FutharkParser::PScremaContext::pSubExp() {
  return getRuleContext<FutharkParser::PSubExpContext>(0);
}

std::vector<tree::TerminalNode *> FutharkParser::PScremaContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::PScremaContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}

tree::TerminalNode* FutharkParser::PScremaContext::OPEN_BRACKET() {
  return getToken(FutharkParser::OPEN_BRACKET, 0);
}

tree::TerminalNode* FutharkParser::PScremaContext::CLOSE_BRACKET() {
  return getToken(FutharkParser::CLOSE_BRACKET, 0);
}

std::vector<tree::TerminalNode *> FutharkParser::PScremaContext::ID() {
  return getTokens(FutharkParser::ID);
}

tree::TerminalNode* FutharkParser::PScremaContext::ID(size_t i) {
  return getToken(FutharkParser::ID, i);
}


size_t FutharkParser::PScremaContext::getRuleIndex() const {
  return FutharkParser::RulePScrema;
}

void FutharkParser::PScremaContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPScrema(this);
}

void FutharkParser::PScremaContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPScrema(this);
}

FutharkParser::PScremaContext* FutharkParser::pScrema() {
  PScremaContext *_localctx = _tracker.createInstance<PScremaContext>(_ctx, getState());
  enterRule(_localctx, 34, FutharkParser::RulePScrema);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(253);
    pSubExp();
    setState(254);
    match(FutharkParser::COMMA);
    setState(255);
    match(FutharkParser::OPEN_BRACKET);
    setState(257);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == FutharkParser::ID) {
      setState(256);
      match(FutharkParser::ID);
    }
    setState(263);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == FutharkParser::COMMA) {
      setState(259);
      match(FutharkParser::COMMA);
      setState(260);
      match(FutharkParser::ID);
      setState(265);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(266);
    match(FutharkParser::CLOSE_BRACKET);
    setState(267);
    match(FutharkParser::COMMA);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PMapFormContext ------------------------------------------------------------------

FutharkParser::PMapFormContext::PMapFormContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

FutharkParser::PLambdaContext* FutharkParser::PMapFormContext::pLambda() {
  return getRuleContext<FutharkParser::PLambdaContext>(0);
}


size_t FutharkParser::PMapFormContext::getRuleIndex() const {
  return FutharkParser::RulePMapForm;
}

void FutharkParser::PMapFormContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPMapForm(this);
}

void FutharkParser::PMapFormContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPMapForm(this);
}

FutharkParser::PMapFormContext* FutharkParser::pMapForm() {
  PMapFormContext *_localctx = _tracker.createInstance<PMapFormContext>(_ctx, getState());
  enterRule(_localctx, 36, FutharkParser::RulePMapForm);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(269);
    pLambda();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PRedomapFormContext ------------------------------------------------------------------

FutharkParser::PRedomapFormContext::PRedomapFormContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

FutharkParser::PLambdaContext* FutharkParser::PRedomapFormContext::pLambda() {
  return getRuleContext<FutharkParser::PLambdaContext>(0);
}

std::vector<tree::TerminalNode *> FutharkParser::PRedomapFormContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::PRedomapFormContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}

tree::TerminalNode* FutharkParser::PRedomapFormContext::OPEN_BRACKET() {
  return getToken(FutharkParser::OPEN_BRACKET, 0);
}

tree::TerminalNode* FutharkParser::PRedomapFormContext::CLOSE_BRACKET() {
  return getToken(FutharkParser::CLOSE_BRACKET, 0);
}

std::vector<FutharkParser::PReduceContext *> FutharkParser::PRedomapFormContext::pReduce() {
  return getRuleContexts<FutharkParser::PReduceContext>();
}

FutharkParser::PReduceContext* FutharkParser::PRedomapFormContext::pReduce(size_t i) {
  return getRuleContext<FutharkParser::PReduceContext>(i);
}


size_t FutharkParser::PRedomapFormContext::getRuleIndex() const {
  return FutharkParser::RulePRedomapForm;
}

void FutharkParser::PRedomapFormContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPRedomapForm(this);
}

void FutharkParser::PRedomapFormContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPRedomapForm(this);
}

FutharkParser::PRedomapFormContext* FutharkParser::pRedomapForm() {
  PRedomapFormContext *_localctx = _tracker.createInstance<PRedomapFormContext>(_ctx, getState());
  enterRule(_localctx, 38, FutharkParser::RulePRedomapForm);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(271);
    pLambda();
    setState(272);
    match(FutharkParser::COMMA);
    setState(273);
    match(FutharkParser::OPEN_BRACKET);
    setState(275);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == FutharkParser::T__12

    || _la == FutharkParser::T__13) {
      setState(274);
      pReduce();
    }
    setState(281);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == FutharkParser::COMMA) {
      setState(277);
      match(FutharkParser::COMMA);
      setState(278);
      pReduce();
      setState(283);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(284);
    match(FutharkParser::CLOSE_BRACKET);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PReduceContext ------------------------------------------------------------------

FutharkParser::PReduceContext::PReduceContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

FutharkParser::PCommContext* FutharkParser::PReduceContext::pComm() {
  return getRuleContext<FutharkParser::PCommContext>(0);
}

FutharkParser::PLambdaContext* FutharkParser::PReduceContext::pLambda() {
  return getRuleContext<FutharkParser::PLambdaContext>(0);
}

std::vector<tree::TerminalNode *> FutharkParser::PReduceContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::PReduceContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}

tree::TerminalNode* FutharkParser::PReduceContext::OPEN_BRACKET() {
  return getToken(FutharkParser::OPEN_BRACKET, 0);
}

tree::TerminalNode* FutharkParser::PReduceContext::CLOSE_BRACKET() {
  return getToken(FutharkParser::CLOSE_BRACKET, 0);
}

std::vector<FutharkParser::PSubExpContext *> FutharkParser::PReduceContext::pSubExp() {
  return getRuleContexts<FutharkParser::PSubExpContext>();
}

FutharkParser::PSubExpContext* FutharkParser::PReduceContext::pSubExp(size_t i) {
  return getRuleContext<FutharkParser::PSubExpContext>(i);
}


size_t FutharkParser::PReduceContext::getRuleIndex() const {
  return FutharkParser::RulePReduce;
}

void FutharkParser::PReduceContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPReduce(this);
}

void FutharkParser::PReduceContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPReduce(this);
}

FutharkParser::PReduceContext* FutharkParser::pReduce() {
  PReduceContext *_localctx = _tracker.createInstance<PReduceContext>(_ctx, getState());
  enterRule(_localctx, 40, FutharkParser::RulePReduce);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(286);
    pComm();
    setState(287);
    pLambda();
    setState(288);
    match(FutharkParser::COMMA);
    setState(289);
    match(FutharkParser::OPEN_BRACKET);
    setState(291);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (((((_la - 27) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 27)) & 171798691841) != 0)) {
      setState(290);
      pSubExp();
    }
    setState(297);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == FutharkParser::COMMA) {
      setState(293);
      match(FutharkParser::COMMA);
      setState(294);
      pSubExp();
      setState(299);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(300);
    match(FutharkParser::CLOSE_BRACKET);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PCommContext ------------------------------------------------------------------

FutharkParser::PCommContext::PCommContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t FutharkParser::PCommContext::getRuleIndex() const {
  return FutharkParser::RulePComm;
}

void FutharkParser::PCommContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPComm(this);
}

void FutharkParser::PCommContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPComm(this);
}

FutharkParser::PCommContext* FutharkParser::pComm() {
  PCommContext *_localctx = _tracker.createInstance<PCommContext>(_ctx, getState());
  enterRule(_localctx, 42, FutharkParser::RulePComm);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(303);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == FutharkParser::T__12) {
      setState(302);
      match(FutharkParser::T__12);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PLambdaContext ------------------------------------------------------------------

FutharkParser::PLambdaContext::PLambdaContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FutharkParser::PLambdaContext::OPEN_BRACKET() {
  return getToken(FutharkParser::OPEN_BRACKET, 0);
}

tree::TerminalNode* FutharkParser::PLambdaContext::CLOSE_BRACKET() {
  return getToken(FutharkParser::CLOSE_BRACKET, 0);
}

FutharkParser::PTypesContext* FutharkParser::PLambdaContext::pTypes() {
  return getRuleContext<FutharkParser::PTypesContext>(0);
}

FutharkParser::PBodyContext* FutharkParser::PLambdaContext::pBody() {
  return getRuleContext<FutharkParser::PBodyContext>(0);
}

std::vector<FutharkParser::PLParamContext *> FutharkParser::PLambdaContext::pLParam() {
  return getRuleContexts<FutharkParser::PLParamContext>();
}

FutharkParser::PLParamContext* FutharkParser::PLambdaContext::pLParam(size_t i) {
  return getRuleContext<FutharkParser::PLParamContext>(i);
}

std::vector<tree::TerminalNode *> FutharkParser::PLambdaContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::PLambdaContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}


size_t FutharkParser::PLambdaContext::getRuleIndex() const {
  return FutharkParser::RulePLambda;
}

void FutharkParser::PLambdaContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPLambda(this);
}

void FutharkParser::PLambdaContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPLambda(this);
}

FutharkParser::PLambdaContext* FutharkParser::pLambda() {
  PLambdaContext *_localctx = _tracker.createInstance<PLambdaContext>(_ctx, getState());
  enterRule(_localctx, 44, FutharkParser::RulePLambda);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(305);
    match(FutharkParser::T__13);
    setState(306);
    match(FutharkParser::OPEN_BRACKET);
    setState(308);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == FutharkParser::ID) {
      setState(307);
      pLParam();
    }
    setState(314);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == FutharkParser::COMMA) {
      setState(310);
      match(FutharkParser::COMMA);
      setState(311);
      pLParam();
      setState(316);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(317);
    match(FutharkParser::CLOSE_BRACKET);
    setState(318);
    match(FutharkParser::T__3);
    setState(319);
    pTypes();
    setState(320);
    match(FutharkParser::T__14);
    setState(321);
    pBody();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PLParamContext ------------------------------------------------------------------

FutharkParser::PLParamContext::PLParamContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FutharkParser::PLParamContext::ID() {
  return getToken(FutharkParser::ID, 0);
}

FutharkParser::PTypeContext* FutharkParser::PLParamContext::pType() {
  return getRuleContext<FutharkParser::PTypeContext>(0);
}


size_t FutharkParser::PLParamContext::getRuleIndex() const {
  return FutharkParser::RulePLParam;
}

void FutharkParser::PLParamContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPLParam(this);
}

void FutharkParser::PLParamContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPLParam(this);
}

FutharkParser::PLParamContext* FutharkParser::pLParam() {
  PLParamContext *_localctx = _tracker.createInstance<PLParamContext>(_ctx, getState());
  enterRule(_localctx, 46, FutharkParser::RulePLParam);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(323);
    match(FutharkParser::ID);
    setState(324);
    match(FutharkParser::T__3);
    setState(325);
    pType();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PApplyContext ------------------------------------------------------------------

FutharkParser::PApplyContext::PApplyContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FutharkParser::PApplyContext::ID() {
  return getToken(FutharkParser::ID, 0);
}

tree::TerminalNode* FutharkParser::PApplyContext::LPARAM() {
  return getToken(FutharkParser::LPARAM, 0);
}

tree::TerminalNode* FutharkParser::PApplyContext::RPARAM() {
  return getToken(FutharkParser::RPARAM, 0);
}

FutharkParser::PRetTypesContext* FutharkParser::PApplyContext::pRetTypes() {
  return getRuleContext<FutharkParser::PRetTypesContext>(0);
}

std::vector<FutharkParser::PSubExpContext *> FutharkParser::PApplyContext::pSubExp() {
  return getRuleContexts<FutharkParser::PSubExpContext>();
}

FutharkParser::PSubExpContext* FutharkParser::PApplyContext::pSubExp(size_t i) {
  return getRuleContext<FutharkParser::PSubExpContext>(i);
}

std::vector<tree::TerminalNode *> FutharkParser::PApplyContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::PApplyContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}


size_t FutharkParser::PApplyContext::getRuleIndex() const {
  return FutharkParser::RulePApply;
}

void FutharkParser::PApplyContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPApply(this);
}

void FutharkParser::PApplyContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPApply(this);
}

FutharkParser::PApplyContext* FutharkParser::pApply() {
  PApplyContext *_localctx = _tracker.createInstance<PApplyContext>(_ctx, getState());
  enterRule(_localctx, 48, FutharkParser::RulePApply);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(327);
    match(FutharkParser::T__15);
    setState(328);
    match(FutharkParser::ID);
    setState(329);
    match(FutharkParser::LPARAM);
    setState(331);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (((((_la - 27) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 27)) & 171798691841) != 0)) {
      setState(330);
      pSubExp();
    }
    setState(337);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == FutharkParser::COMMA) {
      setState(333);
      match(FutharkParser::COMMA);
      setState(334);
      pSubExp();
      setState(339);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(340);
    match(FutharkParser::RPARAM);
    setState(341);
    match(FutharkParser::T__3);
    setState(342);
    pRetTypes();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PArgContext ------------------------------------------------------------------

FutharkParser::PArgContext::PArgContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

FutharkParser::PSubExpContext* FutharkParser::PArgContext::pSubExp() {
  return getRuleContext<FutharkParser::PSubExpContext>(0);
}


size_t FutharkParser::PArgContext::getRuleIndex() const {
  return FutharkParser::RulePArg;
}

void FutharkParser::PArgContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPArg(this);
}

void FutharkParser::PArgContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPArg(this);
}

FutharkParser::PArgContext* FutharkParser::pArg() {
  PArgContext *_localctx = _tracker.createInstance<PArgContext>(_ctx, getState());
  enterRule(_localctx, 50, FutharkParser::RulePArg);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(344);
    pSubExp();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PArrayLitContext ------------------------------------------------------------------

FutharkParser::PArrayLitContext::PArrayLitContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<FutharkParser::PSubExpContext *> FutharkParser::PArrayLitContext::pSubExp() {
  return getRuleContexts<FutharkParser::PSubExpContext>();
}

FutharkParser::PSubExpContext* FutharkParser::PArrayLitContext::pSubExp(size_t i) {
  return getRuleContext<FutharkParser::PSubExpContext>(i);
}

std::vector<tree::TerminalNode *> FutharkParser::PArrayLitContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::PArrayLitContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}


size_t FutharkParser::PArrayLitContext::getRuleIndex() const {
  return FutharkParser::RulePArrayLit;
}

void FutharkParser::PArrayLitContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPArrayLit(this);
}

void FutharkParser::PArrayLitContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPArrayLit(this);
}

FutharkParser::PArrayLitContext* FutharkParser::pArrayLit() {
  PArrayLitContext *_localctx = _tracker.createInstance<PArrayLitContext>(_ctx, getState());
  enterRule(_localctx, 52, FutharkParser::RulePArrayLit);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(346);
    match(FutharkParser::T__6);
    setState(347);
    pSubExp();
    setState(352);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == FutharkParser::COMMA) {
      setState(348);
      match(FutharkParser::COMMA);
      setState(349);
      pSubExp();
      setState(354);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(355);
    match(FutharkParser::T__7);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PBasicOpContext ------------------------------------------------------------------

FutharkParser::PBasicOpContext::PBasicOpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t FutharkParser::PBasicOpContext::getRuleIndex() const {
  return FutharkParser::RulePBasicOp;
}

void FutharkParser::PBasicOpContext::copyFrom(PBasicOpContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- BasicOpArrayLitContext ------------------------------------------------------------------

FutharkParser::PArrayLitContext* FutharkParser::BasicOpArrayLitContext::pArrayLit() {
  return getRuleContext<FutharkParser::PArrayLitContext>(0);
}

FutharkParser::PTypeContext* FutharkParser::BasicOpArrayLitContext::pType() {
  return getRuleContext<FutharkParser::PTypeContext>(0);
}

FutharkParser::BasicOpArrayLitContext::BasicOpArrayLitContext(PBasicOpContext *ctx) { copyFrom(ctx); }

void FutharkParser::BasicOpArrayLitContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBasicOpArrayLit(this);
}
void FutharkParser::BasicOpArrayLitContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBasicOpArrayLit(this);
}
//----------------- BasicOpConcatContext ------------------------------------------------------------------

tree::TerminalNode* FutharkParser::BasicOpConcatContext::CONCAT() {
  return getToken(FutharkParser::CONCAT, 0);
}

tree::TerminalNode* FutharkParser::BasicOpConcatContext::LPARAM() {
  return getToken(FutharkParser::LPARAM, 0);
}

std::vector<FutharkParser::PSubExpContext *> FutharkParser::BasicOpConcatContext::pSubExp() {
  return getRuleContexts<FutharkParser::PSubExpContext>();
}

FutharkParser::PSubExpContext* FutharkParser::BasicOpConcatContext::pSubExp(size_t i) {
  return getRuleContext<FutharkParser::PSubExpContext>(i);
}

std::vector<tree::TerminalNode *> FutharkParser::BasicOpConcatContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::BasicOpConcatContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}

tree::TerminalNode* FutharkParser::BasicOpConcatContext::RPARAM() {
  return getToken(FutharkParser::RPARAM, 0);
}

FutharkParser::BasicOpConcatContext::BasicOpConcatContext(PBasicOpContext *ctx) { copyFrom(ctx); }

void FutharkParser::BasicOpConcatContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBasicOpConcat(this);
}
void FutharkParser::BasicOpConcatContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBasicOpConcat(this);
}
//----------------- BasicOpConvContext ------------------------------------------------------------------

FutharkParser::PConvOpContext* FutharkParser::BasicOpConvContext::pConvOp() {
  return getRuleContext<FutharkParser::PConvOpContext>(0);
}

FutharkParser::BasicOpConvContext::BasicOpConvContext(PBasicOpContext *ctx) { copyFrom(ctx); }

void FutharkParser::BasicOpConvContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBasicOpConv(this);
}
void FutharkParser::BasicOpConvContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBasicOpConv(this);
}
//----------------- BasicOpReplicateContext ------------------------------------------------------------------

tree::TerminalNode* FutharkParser::BasicOpReplicateContext::LPARAM() {
  return getToken(FutharkParser::LPARAM, 0);
}

FutharkParser::PExtShapeContext* FutharkParser::BasicOpReplicateContext::pExtShape() {
  return getRuleContext<FutharkParser::PExtShapeContext>(0);
}

tree::TerminalNode* FutharkParser::BasicOpReplicateContext::COMMA() {
  return getToken(FutharkParser::COMMA, 0);
}

FutharkParser::PSubExpContext* FutharkParser::BasicOpReplicateContext::pSubExp() {
  return getRuleContext<FutharkParser::PSubExpContext>(0);
}

tree::TerminalNode* FutharkParser::BasicOpReplicateContext::RPARAM() {
  return getToken(FutharkParser::RPARAM, 0);
}

FutharkParser::BasicOpReplicateContext::BasicOpReplicateContext(PBasicOpContext *ctx) { copyFrom(ctx); }

void FutharkParser::BasicOpReplicateContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBasicOpReplicate(this);
}
void FutharkParser::BasicOpReplicateContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBasicOpReplicate(this);
}
//----------------- BasicOpIotaContext ------------------------------------------------------------------

tree::TerminalNode* FutharkParser::BasicOpIotaContext::IOTA() {
  return getToken(FutharkParser::IOTA, 0);
}

tree::TerminalNode* FutharkParser::BasicOpIotaContext::LPARAM() {
  return getToken(FutharkParser::LPARAM, 0);
}

std::vector<FutharkParser::PSubExpContext *> FutharkParser::BasicOpIotaContext::pSubExp() {
  return getRuleContexts<FutharkParser::PSubExpContext>();
}

FutharkParser::PSubExpContext* FutharkParser::BasicOpIotaContext::pSubExp(size_t i) {
  return getRuleContext<FutharkParser::PSubExpContext>(i);
}

std::vector<tree::TerminalNode *> FutharkParser::BasicOpIotaContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::BasicOpIotaContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}

tree::TerminalNode* FutharkParser::BasicOpIotaContext::RPARAM() {
  return getToken(FutharkParser::RPARAM, 0);
}

FutharkParser::BasicOpIotaContext::BasicOpIotaContext(PBasicOpContext *ctx) { copyFrom(ctx); }

void FutharkParser::BasicOpIotaContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBasicOpIota(this);
}
void FutharkParser::BasicOpIotaContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBasicOpIota(this);
}
//----------------- BasicOpRearrangeContext ------------------------------------------------------------------

std::vector<tree::TerminalNode *> FutharkParser::BasicOpRearrangeContext::LPARAM() {
  return getTokens(FutharkParser::LPARAM);
}

tree::TerminalNode* FutharkParser::BasicOpRearrangeContext::LPARAM(size_t i) {
  return getToken(FutharkParser::LPARAM, i);
}

FutharkParser::PSubExpContext* FutharkParser::BasicOpRearrangeContext::pSubExp() {
  return getRuleContext<FutharkParser::PSubExpContext>(0);
}

std::vector<tree::TerminalNode *> FutharkParser::BasicOpRearrangeContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::BasicOpRearrangeContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}

std::vector<tree::TerminalNode *> FutharkParser::BasicOpRearrangeContext::NUMBER() {
  return getTokens(FutharkParser::NUMBER);
}

tree::TerminalNode* FutharkParser::BasicOpRearrangeContext::NUMBER(size_t i) {
  return getToken(FutharkParser::NUMBER, i);
}

std::vector<tree::TerminalNode *> FutharkParser::BasicOpRearrangeContext::RPARAM() {
  return getTokens(FutharkParser::RPARAM);
}

tree::TerminalNode* FutharkParser::BasicOpRearrangeContext::RPARAM(size_t i) {
  return getToken(FutharkParser::RPARAM, i);
}

FutharkParser::BasicOpRearrangeContext::BasicOpRearrangeContext(PBasicOpContext *ctx) { copyFrom(ctx); }

void FutharkParser::BasicOpRearrangeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBasicOpRearrange(this);
}
void FutharkParser::BasicOpRearrangeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBasicOpRearrange(this);
}
//----------------- BasicOpFlatIndexContext ------------------------------------------------------------------

std::vector<FutharkParser::PSubExpContext *> FutharkParser::BasicOpFlatIndexContext::pSubExp() {
  return getRuleContexts<FutharkParser::PSubExpContext>();
}

FutharkParser::PSubExpContext* FutharkParser::BasicOpFlatIndexContext::pSubExp(size_t i) {
  return getRuleContext<FutharkParser::PSubExpContext>(i);
}

std::vector<tree::TerminalNode *> FutharkParser::BasicOpFlatIndexContext::COMMA() {
  return getTokens(FutharkParser::COMMA);
}

tree::TerminalNode* FutharkParser::BasicOpFlatIndexContext::COMMA(size_t i) {
  return getToken(FutharkParser::COMMA, i);
}

FutharkParser::BasicOpFlatIndexContext::BasicOpFlatIndexContext(PBasicOpContext *ctx) { copyFrom(ctx); }

void FutharkParser::BasicOpFlatIndexContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBasicOpFlatIndex(this);
}
void FutharkParser::BasicOpFlatIndexContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBasicOpFlatIndex(this);
}
//----------------- BasicOpBinaryContext ------------------------------------------------------------------

FutharkParser::PBinOpContext* FutharkParser::BasicOpBinaryContext::pBinOp() {
  return getRuleContext<FutharkParser::PBinOpContext>(0);
}

FutharkParser::BasicOpBinaryContext::BasicOpBinaryContext(PBasicOpContext *ctx) { copyFrom(ctx); }

void FutharkParser::BasicOpBinaryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBasicOpBinary(this);
}
void FutharkParser::BasicOpBinaryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBasicOpBinary(this);
}
//----------------- BasicOpReshapeContext ------------------------------------------------------------------

std::vector<tree::TerminalNode *> FutharkParser::BasicOpReshapeContext::LPARAM() {
  return getTokens(FutharkParser::LPARAM);
}

tree::TerminalNode* FutharkParser::BasicOpReshapeContext::LPARAM(size_t i) {
  return getToken(FutharkParser::LPARAM, i);
}

FutharkParser::PSubExpContext* FutharkParser::BasicOpReshapeContext::pSubExp() {
  return getRuleContext<FutharkParser::PSubExpContext>(0);
}

tree::TerminalNode* FutharkParser::BasicOpReshapeContext::COMMA() {
  return getToken(FutharkParser::COMMA, 0);
}

std::vector<FutharkParser::PExtShapeContext *> FutharkParser::BasicOpReshapeContext::pExtShape() {
  return getRuleContexts<FutharkParser::PExtShapeContext>();
}

FutharkParser::PExtShapeContext* FutharkParser::BasicOpReshapeContext::pExtShape(size_t i) {
  return getRuleContext<FutharkParser::PExtShapeContext>(i);
}

std::vector<tree::TerminalNode *> FutharkParser::BasicOpReshapeContext::RPARAM() {
  return getTokens(FutharkParser::RPARAM);
}

tree::TerminalNode* FutharkParser::BasicOpReshapeContext::RPARAM(size_t i) {
  return getToken(FutharkParser::RPARAM, i);
}

std::vector<tree::TerminalNode *> FutharkParser::BasicOpReshapeContext::NUMBER() {
  return getTokens(FutharkParser::NUMBER);
}

tree::TerminalNode* FutharkParser::BasicOpReshapeContext::NUMBER(size_t i) {
  return getToken(FutharkParser::NUMBER, i);
}

FutharkParser::BasicOpReshapeContext::BasicOpReshapeContext(PBasicOpContext *ctx) { copyFrom(ctx); }

void FutharkParser::BasicOpReshapeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBasicOpReshape(this);
}
void FutharkParser::BasicOpReshapeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBasicOpReshape(this);
}
FutharkParser::PBasicOpContext* FutharkParser::pBasicOp() {
  PBasicOpContext *_localctx = _tracker.createInstance<PBasicOpContext>(_ctx, getState());
  enterRule(_localctx, 54, FutharkParser::RulePBasicOp);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(432);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case FutharkParser::T__16: {
        _localctx = _tracker.createInstance<FutharkParser::BasicOpReplicateContext>(_localctx);
        enterOuterAlt(_localctx, 1);
        setState(357);
        match(FutharkParser::T__16);
        setState(358);
        match(FutharkParser::LPARAM);
        setState(359);
        pExtShape();
        setState(360);
        match(FutharkParser::COMMA);
        setState(361);
        pSubExp();
        setState(362);
        match(FutharkParser::RPARAM);
        break;
      }

      case FutharkParser::T__6: {
        _localctx = _tracker.createInstance<FutharkParser::BasicOpArrayLitContext>(_localctx);
        enterOuterAlt(_localctx, 2);
        setState(364);
        pArrayLit();
        setState(365);
        match(FutharkParser::T__3);
        setState(366);
        pType();
        break;
      }

      case FutharkParser::NUMBER:
      case FutharkParser::FLOAT:
      case FutharkParser::ID: {
        _localctx = _tracker.createInstance<FutharkParser::BasicOpFlatIndexContext>(_localctx);
        enterOuterAlt(_localctx, 3);
        setState(368);
        pSubExp();
        setState(369);
        match(FutharkParser::T__6);
        setState(370);
        pSubExp();
        setState(375);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == FutharkParser::COMMA) {
          setState(371);
          match(FutharkParser::COMMA);
          setState(372);
          pSubExp();
          setState(377);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(378);
        match(FutharkParser::T__7);
        break;
      }

      case FutharkParser::CONCAT: {
        _localctx = _tracker.createInstance<FutharkParser::BasicOpConcatContext>(_localctx);
        enterOuterAlt(_localctx, 4);
        setState(380);
        match(FutharkParser::CONCAT);
        setState(381);
        match(FutharkParser::LPARAM);
        setState(382);
        pSubExp();
        setState(383);
        match(FutharkParser::COMMA);
        setState(384);
        pSubExp();
        setState(385);
        match(FutharkParser::COMMA);
        setState(386);
        pSubExp();
        setState(387);
        match(FutharkParser::RPARAM);
        break;
      }

      case FutharkParser::IOTA: {
        _localctx = _tracker.createInstance<FutharkParser::BasicOpIotaContext>(_localctx);
        enterOuterAlt(_localctx, 5);
        setState(389);
        match(FutharkParser::IOTA);
        setState(390);
        match(FutharkParser::LPARAM);
        setState(391);
        pSubExp();
        setState(392);
        match(FutharkParser::COMMA);
        setState(393);
        pSubExp();
        setState(394);
        match(FutharkParser::COMMA);
        setState(395);
        pSubExp();
        setState(396);
        match(FutharkParser::RPARAM);
        break;
      }

      case FutharkParser::ADD:
      case FutharkParser::FADD:
      case FutharkParser::SUB:
      case FutharkParser::FSUB:
      case FutharkParser::MUL:
      case FutharkParser::FMUL:
      case FutharkParser::UDIV:
      case FutharkParser::UDIVUP:
      case FutharkParser::SDIV:
      case FutharkParser::SDIVUP:
      case FutharkParser::FDIV:
      case FutharkParser::FMOD:
      case FutharkParser::UMOD:
      case FutharkParser::SMOD:
      case FutharkParser::SQUOT:
      case FutharkParser::SREM:
      case FutharkParser::SMIN:
      case FutharkParser::UMIN:
      case FutharkParser::FMIN:
      case FutharkParser::SMAX:
      case FutharkParser::UMAX:
      case FutharkParser::FMAX:
      case FutharkParser::SHL:
      case FutharkParser::LSHR:
      case FutharkParser::ASHR:
      case FutharkParser::AND:
      case FutharkParser::OR:
      case FutharkParser::XOR:
      case FutharkParser::POW:
      case FutharkParser::FPOW:
      case FutharkParser::LOGAND:
      case FutharkParser::LOGOR: {
        _localctx = _tracker.createInstance<FutharkParser::BasicOpBinaryContext>(_localctx);
        enterOuterAlt(_localctx, 6);
        setState(398);
        pBinOp();
        break;
      }

      case FutharkParser::T__22: {
        _localctx = _tracker.createInstance<FutharkParser::BasicOpConvContext>(_localctx);
        enterOuterAlt(_localctx, 7);
        setState(399);
        pConvOp();
        break;
      }

      case FutharkParser::T__17: {
        _localctx = _tracker.createInstance<FutharkParser::BasicOpReshapeContext>(_localctx);
        enterOuterAlt(_localctx, 8);
        setState(400);
        match(FutharkParser::T__17);
        setState(401);
        match(FutharkParser::LPARAM);
        setState(402);
        pSubExp();
        setState(403);
        match(FutharkParser::COMMA);
        setState(404);
        match(FutharkParser::LPARAM);

        setState(405);
        match(FutharkParser::NUMBER);
        setState(406);
        match(FutharkParser::T__18);
        setState(407);
        match(FutharkParser::NUMBER);
        setState(408);
        match(FutharkParser::T__19);
        setState(409);
        pExtShape();
        setState(410);
        match(FutharkParser::T__20);
        setState(412);
        pExtShape();
        setState(413);
        match(FutharkParser::RPARAM);
        setState(414);
        match(FutharkParser::RPARAM);
        break;
      }

      case FutharkParser::T__21: {
        _localctx = _tracker.createInstance<FutharkParser::BasicOpRearrangeContext>(_localctx);
        enterOuterAlt(_localctx, 9);
        setState(416);
        match(FutharkParser::T__21);
        setState(417);
        match(FutharkParser::LPARAM);
        setState(418);
        pSubExp();
        setState(419);
        match(FutharkParser::COMMA);
        setState(420);
        match(FutharkParser::LPARAM);
        setState(421);
        match(FutharkParser::NUMBER);
        setState(426);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == FutharkParser::COMMA) {
          setState(422);
          match(FutharkParser::COMMA);
          setState(423);
          match(FutharkParser::NUMBER);
          setState(428);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(429);
        match(FutharkParser::RPARAM);
        setState(430);
        match(FutharkParser::RPARAM);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PBinOpContext ------------------------------------------------------------------

FutharkParser::PBinOpContext::PBinOpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

FutharkParser::BinaryOpcodeContext* FutharkParser::PBinOpContext::binaryOpcode() {
  return getRuleContext<FutharkParser::BinaryOpcodeContext>(0);
}

tree::TerminalNode* FutharkParser::PBinOpContext::LPARAM() {
  return getToken(FutharkParser::LPARAM, 0);
}

std::vector<FutharkParser::PSubExpContext *> FutharkParser::PBinOpContext::pSubExp() {
  return getRuleContexts<FutharkParser::PSubExpContext>();
}

FutharkParser::PSubExpContext* FutharkParser::PBinOpContext::pSubExp(size_t i) {
  return getRuleContext<FutharkParser::PSubExpContext>(i);
}

tree::TerminalNode* FutharkParser::PBinOpContext::COMMA() {
  return getToken(FutharkParser::COMMA, 0);
}

tree::TerminalNode* FutharkParser::PBinOpContext::RPARAM() {
  return getToken(FutharkParser::RPARAM, 0);
}


size_t FutharkParser::PBinOpContext::getRuleIndex() const {
  return FutharkParser::RulePBinOp;
}

void FutharkParser::PBinOpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPBinOp(this);
}

void FutharkParser::PBinOpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPBinOp(this);
}

FutharkParser::PBinOpContext* FutharkParser::pBinOp() {
  PBinOpContext *_localctx = _tracker.createInstance<PBinOpContext>(_ctx, getState());
  enterRule(_localctx, 56, FutharkParser::RulePBinOp);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(434);
    binaryOpcode();
    setState(435);
    match(FutharkParser::LPARAM);
    setState(436);
    pSubExp();
    setState(437);
    match(FutharkParser::COMMA);
    setState(438);
    pSubExp();
    setState(439);
    match(FutharkParser::RPARAM);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BinaryOpcodeContext ------------------------------------------------------------------

FutharkParser::BinaryOpcodeContext::BinaryOpcodeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::ADD() {
  return getToken(FutharkParser::ADD, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::FADD() {
  return getToken(FutharkParser::FADD, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::SUB() {
  return getToken(FutharkParser::SUB, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::FSUB() {
  return getToken(FutharkParser::FSUB, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::MUL() {
  return getToken(FutharkParser::MUL, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::FMUL() {
  return getToken(FutharkParser::FMUL, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::UDIV() {
  return getToken(FutharkParser::UDIV, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::UDIVUP() {
  return getToken(FutharkParser::UDIVUP, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::SDIV() {
  return getToken(FutharkParser::SDIV, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::SDIVUP() {
  return getToken(FutharkParser::SDIVUP, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::FDIV() {
  return getToken(FutharkParser::FDIV, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::FMOD() {
  return getToken(FutharkParser::FMOD, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::UMOD() {
  return getToken(FutharkParser::UMOD, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::SMOD() {
  return getToken(FutharkParser::SMOD, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::SQUOT() {
  return getToken(FutharkParser::SQUOT, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::SREM() {
  return getToken(FutharkParser::SREM, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::SMIN() {
  return getToken(FutharkParser::SMIN, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::UMIN() {
  return getToken(FutharkParser::UMIN, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::FMIN() {
  return getToken(FutharkParser::FMIN, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::SMAX() {
  return getToken(FutharkParser::SMAX, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::UMAX() {
  return getToken(FutharkParser::UMAX, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::FMAX() {
  return getToken(FutharkParser::FMAX, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::SHL() {
  return getToken(FutharkParser::SHL, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::LSHR() {
  return getToken(FutharkParser::LSHR, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::ASHR() {
  return getToken(FutharkParser::ASHR, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::AND() {
  return getToken(FutharkParser::AND, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::OR() {
  return getToken(FutharkParser::OR, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::XOR() {
  return getToken(FutharkParser::XOR, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::POW() {
  return getToken(FutharkParser::POW, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::FPOW() {
  return getToken(FutharkParser::FPOW, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::LOGAND() {
  return getToken(FutharkParser::LOGAND, 0);
}

tree::TerminalNode* FutharkParser::BinaryOpcodeContext::LOGOR() {
  return getToken(FutharkParser::LOGOR, 0);
}


size_t FutharkParser::BinaryOpcodeContext::getRuleIndex() const {
  return FutharkParser::RuleBinaryOpcode;
}

void FutharkParser::BinaryOpcodeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBinaryOpcode(this);
}

void FutharkParser::BinaryOpcodeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBinaryOpcode(this);
}

FutharkParser::BinaryOpcodeContext* FutharkParser::binaryOpcode() {
  BinaryOpcodeContext *_localctx = _tracker.createInstance<BinaryOpcodeContext>(_ctx, getState());
  enterRule(_localctx, 58, FutharkParser::RuleBinaryOpcode);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(441);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 4611686017353646080) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PConvOpContext ------------------------------------------------------------------

FutharkParser::PConvOpContext::PConvOpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<FutharkParser::PPrimTypeContext *> FutharkParser::PConvOpContext::pPrimType() {
  return getRuleContexts<FutharkParser::PPrimTypeContext>();
}

FutharkParser::PPrimTypeContext* FutharkParser::PConvOpContext::pPrimType(size_t i) {
  return getRuleContext<FutharkParser::PPrimTypeContext>(i);
}

FutharkParser::PSubExpContext* FutharkParser::PConvOpContext::pSubExp() {
  return getRuleContext<FutharkParser::PSubExpContext>(0);
}


size_t FutharkParser::PConvOpContext::getRuleIndex() const {
  return FutharkParser::RulePConvOp;
}

void FutharkParser::PConvOpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPConvOp(this);
}

void FutharkParser::PConvOpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPConvOp(this);
}

FutharkParser::PConvOpContext* FutharkParser::pConvOp() {
  PConvOpContext *_localctx = _tracker.createInstance<PConvOpContext>(_ctx, getState());
  enterRule(_localctx, 60, FutharkParser::RulePConvOp);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(443);
    match(FutharkParser::T__22);
    setState(444);
    pPrimType();
    setState(445);
    pSubExp();
    setState(446);
    match(FutharkParser::T__23);
    setState(447);
    pPrimType();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PSubExpContext ------------------------------------------------------------------

FutharkParser::PSubExpContext::PSubExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t FutharkParser::PSubExpContext::getRuleIndex() const {
  return FutharkParser::RulePSubExp;
}

void FutharkParser::PSubExpContext::copyFrom(PSubExpContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- VarSubExpContext ------------------------------------------------------------------

tree::TerminalNode* FutharkParser::VarSubExpContext::ID() {
  return getToken(FutharkParser::ID, 0);
}

FutharkParser::VarSubExpContext::VarSubExpContext(PSubExpContext *ctx) { copyFrom(ctx); }

void FutharkParser::VarSubExpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVarSubExp(this);
}
void FutharkParser::VarSubExpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVarSubExp(this);
}
//----------------- ConstantSubExpressionContext ------------------------------------------------------------------

FutharkParser::PPrimValueContext* FutharkParser::ConstantSubExpressionContext::pPrimValue() {
  return getRuleContext<FutharkParser::PPrimValueContext>(0);
}

FutharkParser::ConstantSubExpressionContext::ConstantSubExpressionContext(PSubExpContext *ctx) { copyFrom(ctx); }

void FutharkParser::ConstantSubExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterConstantSubExpression(this);
}
void FutharkParser::ConstantSubExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitConstantSubExpression(this);
}
FutharkParser::PSubExpContext* FutharkParser::pSubExp() {
  PSubExpContext *_localctx = _tracker.createInstance<PSubExpContext>(_ctx, getState());
  enterRule(_localctx, 62, FutharkParser::RulePSubExp);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(451);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case FutharkParser::NUMBER:
      case FutharkParser::FLOAT: {
        _localctx = _tracker.createInstance<FutharkParser::ConstantSubExpressionContext>(_localctx);
        enterOuterAlt(_localctx, 1);
        setState(449);
        pPrimValue();
        break;
      }

      case FutharkParser::ID: {
        _localctx = _tracker.createInstance<FutharkParser::VarSubExpContext>(_localctx);
        enterOuterAlt(_localctx, 2);
        setState(450);
        match(FutharkParser::ID);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PPrimValueContext ------------------------------------------------------------------

FutharkParser::PPrimValueContext::PPrimValueContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t FutharkParser::PPrimValueContext::getRuleIndex() const {
  return FutharkParser::RulePPrimValue;
}

void FutharkParser::PPrimValueContext::copyFrom(PPrimValueContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- PrimValueIntegerContext ------------------------------------------------------------------

tree::TerminalNode* FutharkParser::PrimValueIntegerContext::NUMBER() {
  return getToken(FutharkParser::NUMBER, 0);
}

tree::TerminalNode* FutharkParser::PrimValueIntegerContext::INTEGER_TYPE() {
  return getToken(FutharkParser::INTEGER_TYPE, 0);
}

FutharkParser::PrimValueIntegerContext::PrimValueIntegerContext(PPrimValueContext *ctx) { copyFrom(ctx); }

void FutharkParser::PrimValueIntegerContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPrimValueInteger(this);
}
void FutharkParser::PrimValueIntegerContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPrimValueInteger(this);
}
//----------------- PrimValueFloatContext ------------------------------------------------------------------

tree::TerminalNode* FutharkParser::PrimValueFloatContext::FLOAT() {
  return getToken(FutharkParser::FLOAT, 0);
}

tree::TerminalNode* FutharkParser::PrimValueFloatContext::FLOAT_TYPE() {
  return getToken(FutharkParser::FLOAT_TYPE, 0);
}

FutharkParser::PrimValueFloatContext::PrimValueFloatContext(PPrimValueContext *ctx) { copyFrom(ctx); }

void FutharkParser::PrimValueFloatContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPrimValueFloat(this);
}
void FutharkParser::PrimValueFloatContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPrimValueFloat(this);
}
FutharkParser::PPrimValueContext* FutharkParser::pPrimValue() {
  PPrimValueContext *_localctx = _tracker.createInstance<PPrimValueContext>(_ctx, getState());
  enterRule(_localctx, 64, FutharkParser::RulePPrimValue);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(457);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case FutharkParser::NUMBER: {
        _localctx = _tracker.createInstance<FutharkParser::PrimValueIntegerContext>(_localctx);
        enterOuterAlt(_localctx, 1);
        setState(453);
        match(FutharkParser::NUMBER);
        setState(454);
        match(FutharkParser::INTEGER_TYPE);
        break;
      }

      case FutharkParser::FLOAT: {
        _localctx = _tracker.createInstance<FutharkParser::PrimValueFloatContext>(_localctx);
        enterOuterAlt(_localctx, 2);
        setState(455);
        match(FutharkParser::FLOAT);
        setState(456);
        match(FutharkParser::FLOAT_TYPE);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PPrimTypeContext ------------------------------------------------------------------

FutharkParser::PPrimTypeContext::PPrimTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t FutharkParser::PPrimTypeContext::getRuleIndex() const {
  return FutharkParser::RulePPrimType;
}

void FutharkParser::PPrimTypeContext::copyFrom(PPrimTypeContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- PrimTypeIntegerContext ------------------------------------------------------------------

tree::TerminalNode* FutharkParser::PrimTypeIntegerContext::INTEGER_TYPE() {
  return getToken(FutharkParser::INTEGER_TYPE, 0);
}

FutharkParser::PrimTypeIntegerContext::PrimTypeIntegerContext(PPrimTypeContext *ctx) { copyFrom(ctx); }

void FutharkParser::PrimTypeIntegerContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPrimTypeInteger(this);
}
void FutharkParser::PrimTypeIntegerContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPrimTypeInteger(this);
}
//----------------- PrimTypeFloatContext ------------------------------------------------------------------

tree::TerminalNode* FutharkParser::PrimTypeFloatContext::FLOAT_TYPE() {
  return getToken(FutharkParser::FLOAT_TYPE, 0);
}

FutharkParser::PrimTypeFloatContext::PrimTypeFloatContext(PPrimTypeContext *ctx) { copyFrom(ctx); }

void FutharkParser::PrimTypeFloatContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPrimTypeFloat(this);
}
void FutharkParser::PrimTypeFloatContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FutharkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPrimTypeFloat(this);
}
FutharkParser::PPrimTypeContext* FutharkParser::pPrimType() {
  PPrimTypeContext *_localctx = _tracker.createInstance<PPrimTypeContext>(_ctx, getState());
  enterRule(_localctx, 66, FutharkParser::RulePPrimType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(461);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case FutharkParser::INTEGER_TYPE: {
        _localctx = _tracker.createInstance<FutharkParser::PrimTypeIntegerContext>(_localctx);
        enterOuterAlt(_localctx, 1);
        setState(459);
        match(FutharkParser::INTEGER_TYPE);
        break;
      }

      case FutharkParser::FLOAT_TYPE: {
        _localctx = _tracker.createInstance<FutharkParser::PrimTypeFloatContext>(_localctx);
        enterOuterAlt(_localctx, 2);
        setState(460);
        match(FutharkParser::FLOAT_TYPE);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

void FutharkParser::initialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  futharkParserInitialize();
#else
  ::antlr4::internal::call_once(futharkParserOnceFlag, futharkParserInitialize);
#endif
}
