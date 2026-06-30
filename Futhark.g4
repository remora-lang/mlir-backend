grammar Futhark;

root: header? pFunDef+ EOF;

header: 'name_source' '{' NUMBER '}' 'types' '{' '}';

pFunDef: (pEntry? | 'fun') ID LPARAM fParam? (',' fParam)* RPARAM ':' OPEN_BRACKET pType CLOSE_BRACKET '=' OPEN_BRACKET pBody CLOSE_BRACKET;

pEntry: ('entry') LPARAM STRING_LITERAL COMMA OPEN_BRACKET CLOSE_BRACKET COMMA OPEN_BRACKET pType CLOSE_BRACKET RPARAM;

// Empty for now
pEntryPointInput: ;

fParam: ID ':' pType;

// TODO: Aliasing annotations
pRetTypes: '{' pType* (',' pType)* '}';
pType: pExtShape pPrimType #TypeShape
    | pPrimType #TypePrim;
pTypes: '{' pType? (',' pType)* '}';
pRetType: pType;

pExtShape: ('[' pSubExp* ']')+;

pBody: pStm* 'in' OPEN_BRACKET pSubExp (',' pSubExp)* CLOSE_BRACKET #Body
    | OPEN_BRACKET pSubExp (',' pSubExp)* CLOSE_BRACKET #EmptyBody;

pStm: 'let' pPat '=' pExp;

pPat: OPEN_BRACKET pPatElem (pPatElem ',')* CLOSE_BRACKET;

pPatElem: ID ':' pType;

// https://hackage-content.haskell.org/package/futhark-0.25.34/docs/Futhark-IR-Syntax.html#t:Exp
pExp: pApply #ExpApply
    | pBasicOp #ExpBasicOp
    | pSoacOp #ExpSoacOp
    | pSubExp #ExpSubExp;

pSoacOp: 'map' '('  pScrema pMapForm ')' #SoacOpMap
| 'redomap' '('  pScrema pRedomapForm ')' #SoacOpRedomap;

pScrema: pSubExp ',' '{' ID? (',' ID)* '}' ',';

pMapForm: pLambda;

pRedomapForm: pLambda ',' '{' pReduce? (',' pReduce)* '}';

pReduce: pComm pLambda  ',' '{' pSubExp? (',' pSubExp)* '}';

pComm: 'commutative'?;

pLambda: '\\' '{' pLParam? (',' pLParam)* '}' ':' pTypes '->' pBody;

pLParam: ID ':' pType;

pApply: ('apply') ID '(' pSubExp?   (',' pSubExp)* ')' ':' pRetTypes;

pArg: pSubExp;

pArrayLit: '[' pSubExp (',' pSubExp)* ']';

// https://hackage-content.haskell.org/package/futhark-0.25.34/docs/Futhark-IR-Syntax.html#t:BasicOp
pBasicOp: 'replicate' '(' pExtShape ',' pSubExp ')' #BasicOpReplicate
    | pArrayLit ':' pType #BasicOpArrayLit
    | pSubExp '[' pSubExp (',' pSubExp)* ']' #BasicOpFlatIndex
    | CONCAT '(' pSubExp ',' pSubExp ',' pSubExp ')' #BasicOpConcat
    | IOTA '(' pSubExp ',' pSubExp ',' pSubExp ')' #BasicOpIota
    | pBinOp #BasicOpBinary
    | pConvOp #BasicOpConv
    | pCmpOp #BasicOpCompare
    | pUnOp #BasicOpUnary
    | 'reshape' '(' pSubExp ',' '(' (NUMBER '::' NUMBER '=>' pExtShape ';') pExtShape ')' ')'  #BasicOpReshape
    | 'rearrange' '(' pSubExp ',' '(' NUMBER (',' NUMBER)* ')' ')' #BasicOpRearrange
    ;

pBinOp: binaryOpcode LPARAM pSubExp ',' pSubExp RPARAM;

binaryOpcode: ADD | FADD | SUB | FSUB | MUL | FMUL | UDIV | UDIVUP | SDIV | SDIVUP | FDIV | FMOD | UMOD | SMOD | SQUOT | SREM | SMIN | UMIN | FMIN | SMAX | UMAX | FMAX | SHL | LSHR | ASHR | AND | OR | XOR | POW | FPOW | LOGAND | LOGOR;

pCmpOp: compareOpcode LPARAM pSubExp ',' pSubExp RPARAM;
compareOpcode: CMPEQ | CMPULT | CMPULE | CMPSLT | CMPSLE | FCMPLT | FCMPLE | CMPLLT | CMPLLE;

pConvOp
  : CONV_EXT INTEGER_TYPE pSubExp 'to' INTEGER_TYPE
  | CONV_FPCONV FLOAT_TYPE pSubExp 'to' FLOAT_TYPE
  | CONV_F2I FLOAT_TYPE pSubExp 'to' INTEGER_TYPE
  | CONV_I2F INTEGER_TYPE pSubExp 'to' FLOAT_TYPE
  ;

pUnOp: unaryOpcode pSubExp;
unaryOpcode
  : NEGI
  | NEGF
  | COMPLEMENT
  | ABS
  | FABS
  | SSIGNUM
  | USIGNUM
  | FSIGNUM
  ;

pSubExp: pPrimValue #ConstantSubExpression
    | ID #VarSubExp;

pPrimValue: NUMBER INTEGER_TYPE #PrimValueInteger
    | FLOAT FLOAT_TYPE #PrimValueFloat;

pPrimType: INTEGER_TYPE #PrimTypeInteger
 | FLOAT_TYPE #PrimTypeFloat;

INTEGER_TYPE: 'i8' | 'i16' | 'i32' | 'i64';
FLOAT_TYPE: 'f16' | 'f32' | 'f64';

NUMBER: ([0-9]+) | ('0x' ([a-fA-F0-9])+);

// Binary operators
IOTA: 'iota' NUMBER;
CONCAT: 'concat' '@' NUMBER;

ADD: 'add' NUMBER;
FADD: 'fadd' NUMBER;
SUB: 'sub' NUMBER;
FSUB: 'fsub' NUMBER;
MUL: 'mul' NUMBER;
FMUL: 'fmul' NUMBER;
UDIV: 'udiv' NUMBER;
UDIVUP: 'udivup' NUMBER;
SDIV: 'sdiv' NUMBER;
SDIVUP: 'sdivup' NUMBER;
FDIV: 'fdiv' NUMBER;
FMOD: 'fmod' NUMBER;
UMOD: 'umod' NUMBER;
SMOD: 'smod' NUMBER;
SQUOT: 'squot' NUMBER;
SREM: 'srem' NUMBER;
SMIN: 'smin' NUMBER;
UMIN: 'umin' NUMBER;
FMIN: 'fmin' NUMBER;
SMAX: 'smax' NUMBER;
UMAX: 'umax' NUMBER;
FMAX: 'fmax' NUMBER;
SHL: 'shl' NUMBER;
LSHR: 'lshr' NUMBER;
ASHR: 'ashr' NUMBER;
AND: 'and' NUMBER;
OR: 'or' NUMBER;
XOR: 'xor' NUMBER;
POW: 'pow' NUMBER;
FPOW: 'fpow' NUMBER;
LOGAND: 'logand' NUMBER;
LOGOR: 'logor' NUMBER;
// Comparison Operators
CMPEQ: 'eq_i' NUMBER;
FCMPEQ: 'eq_f' NUMBER;
CMPULT: 'ult' NUMBER;
CMPULE: 'ule' NUMBER;
CMPSLT: 'slt' NUMBER;
CMPSLE: 'sle' NUMBER;
FCMPLT: 'flt' NUMBER;
FCMPLE: 'fle' NUMBER;
CMPLLT: 'llt' NUMBER;
CMPLLE: 'lle' NUMBER;
// Coversion Operators
CONV_EXT: 'zext' | 'sext';
CONV_FPCONV: 'fpconv';
CONV_F2I: 'fptoui' | 'fptosi' | 'uitofp' | 'sitofp';
CONV_I2F: 'fptobits' | 'bitstofp';
// Unary Operators
NEGI: 'neg_i' NUMBER;
NEGF: 'neg_f' NUMBER;
COMPLEMENT: 'complement' NUMBER;
ABS: 'abs' NUMBER;
FABS: 'fabs' NUMBER;
SSIGNUM: 'ssignum' NUMBER;
USIGNUM: 'usignum' NUMBER;
FSIGNUM: 'fsignum' NUMBER;

FLOAT   : (DIGIT+ '.' DIGIT+) ;
fragment DIGIT  : [0-9];

STRING_LITERAL: '"' ID '"';

ID : ( [a-zA-Z_+'-'*/%=!<>|&^.#] ) ([a-zA-Z0-9_+\-*/%=!<>|&^.#])*;

LPARAM      : '(';
RPARAM      : ')';

OPEN_BRACKET: '{';
CLOSE_BRACKET: '}';

COMMA       : ',';
BUILTIN     : 'builtin';
RETURN      : 'return';
FUNCTIONS      : 'FUNCTIONS';
RULES      : 'RULES';

WS          : [ \t\r\n]+ -> skip ;
