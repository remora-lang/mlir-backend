grammar Futhark;

root: header? pStm* pFunDef+ EOF;

header: 'name_source' '{' NUMBER '}' 'types' '{' pTypeBind* '}';

pTypeBind: 'type' STRING_LITERAL '=' pTypeExp;

pTypeExp: 'record' '{' pRecordField* '}'
        | 'record_array' pRank STRING_LITERAL '{' pRecordField* '}';

pRecordField: NUMBER ':' pType;

pRank: NUMBER ID;

pFunDef: pAttr* (pEntry? | 'fun') ID LPARAM fParam? (',' fParam)* RPARAM ':' OPEN_BRACKET '*'? pType (',' '*'? pType)* CLOSE_BRACKET '=' OPEN_BRACKET pBody CLOSE_BRACKET;

pAttr: '#' '[' ID ('(' (ID (',' ID)*)? ')')? ']';

pEntry: ('entry') LPARAM STRING_LITERAL COMMA OPEN_BRACKET pEntryPointInput? (COMMA pEntryPointInput)* CLOSE_BRACKET COMMA (OPAQUE STRING_LITERAL | '*'? pType) RPARAM;

pEntryPointInput: ID ':' pType;

fParam: ID ':' pType;

pRetTypes: '{' pType* (',' pType)* '}';
pType: pExtShape pPrimType pAlias? #TypeShape
    | pPrimType pAlias? #TypePrim
    | pExtShape 'unit' pAlias? #TypeUnitArray
    | 'unit' #TypeUnit
    | 'acc' '(' ID ',' pExtShape ',' pTypes ')' #TypeAcc;

pAlias: '#' '(' pAliasSet (',' pAliasSet)* ')';
pAliasSet: '[' (NUMBER (',' NUMBER)*)? ']';
pTypes: '{' pType? (',' pType)* '}';
pRetType: pType;

pExtShape: ('[' pSubExp* ']')+;

pBody: pStm* 'in' OPEN_BRACKET pSubExp (',' pSubExp)* CLOSE_BRACKET #Body
    | OPEN_BRACKET pSubExp (',' pSubExp)* CLOSE_BRACKET #EmptyBody;

pStm: 'let' pPat '=' pCerts? pExp;

pCerts: '#' '{' (ID (',' ID)*)? '}';

pPat: OPEN_BRACKET pPatElem (',' pPatElem)* CLOSE_BRACKET;

pPatElem: ID ':' pType;

// https://hackage-content.haskell.org/package/futhark-0.25.34/docs/Futhark-IR-Syntax.html#t:Exp
pExp: pApply #ExpApply
    | pBasicOp #ExpBasicOp
    | pSegOp #ExpSegOp
    | pSizeOp #ExpSizeOp
    | 'if' pSubExp 'then' pCaseBody 'else' pCaseBody ':' pTypes #ExpIf
    | 'gpu' ':' pTypes pCaseBody #ExpGpuBody
    | pWithAcc #ExpWithAcc
    | pSubExp #ExpSubExp;

// A branch/kernel body delimited by braces, as printed for `if` arms and
// `gpu` bodies: either `{ stms in {results} }` or just `{ results }`.
pCaseBody: OPEN_BRACKET pStm* 'in' OPEN_BRACKET pSubExp (',' pSubExp)* CLOSE_BRACKET CLOSE_BRACKET #CaseBodyFull
         | OPEN_BRACKET pSubExp (',' pSubExp)* CLOSE_BRACKET #CaseBodyResult;

pSizeOp: 'get_size' '(' ID ',' ID ')';

pWithAcc: 'with_acc' '(' '{' pWithAccInput (',' pWithAccInput)* '}' ',' pLambda ')';

pWithAccInput: '(' pExtShape ',' '{' ID (',' ID)* '}' (',' pWithAccOp)? ')';

pWithAccOp: '(' pLambda ',' '{' (pSubExp (',' pSubExp)*)? '}' ')';

// This is specialised to a certain form of seg ops.
pSegOp : 'segmap' '(' 'thread' ';' ';' ')' pSegSpace ':' pTypes '{' pKernelBody '}' #SegMap
       | 'segred' '(' 'thread' ';' ';' ')' pSegSpace ':' pTypes '{' pKernelBody '}' '(' pSegBinOp (',' pSegBinOp)* ')' #SegRed
       | 'segscan' '(' 'thread' ';' ';' ')' pSegSpace ':' pTypes '{' pKernelBody '}' '(' pSegBinOp (',' pSegBinOp)* ')' pSegPostOp #SegScan
       ;

// Futhark's SegBinOp: {neutrals}, shape, [commutative] lambda.
pSegBinOp: '{' (pSubExp (',' pSubExp)*)? '}' ',' pExtShape? ',' pComm pLambda;

pSegPostOp: pLambda;

pComm: 'commutative'?;

pKernelBody: pStm* 'return' '{' 'returns' pSubExp (',' 'returns' pSubExp)* '}' #KernelBody
    | '{' 'return' pSubExp (',' 'returns' pSubExp)* '}' #EmptyKernelBody;


pSegSpace: '(' ID '<' pSubExp (',' ID '<' pSubExp)* ')' '(' '~' ID ')';

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
    | pCmpOp #BasicOpCmp
    | pConvOp #BasicOpConv
    | 'reshape' '(' pSubExp ',' '(' (NUMBER '::' NUMBER '=>' pExtShape ';') pExtShape ')' ')'  #BasicOpReshape
    | 'rearrange' '(' pSubExp ',' '(' NUMBER (',' NUMBER)* ')' ')' #BasicOpRearrange
    | 'assert' '(' pSubExp ',' '{' pErrPart (',' pErrPart)* '}' ')' #BasicOpAssert
    | 'scratch' '(' pPrimType (',' pSubExp)* ')' #BasicOpScratch
    | 'update_acc' '(' ID ',' '{' pSubExp '}' ',' '{' pSubExp '}' ')' #BasicOpUpdateAcc
    ;

// An interpolated assertion error message: literal chunks interleaved with
// typed subexpressions, e.g. {"a[", i : i64, "]"}.
pErrPart: STRING_LITERAL | pSubExp ':' pPrimType;

pBinOp: binaryOpcode LPARAM pSubExp ',' pSubExp RPARAM;

pCmpOp: cmpOpcode LPARAM pSubExp ',' pSubExp RPARAM;

cmpOpcode: CMPEQ | CMPSLT | CMPSLE | CMPULT | CMPULE;

binaryOpcode: ADD | ADDNW | FADD | SUB | SUBNW | FSUB | MUL | MULNW | FMUL | UDIV | UDIVUP | SDIV | SDIVUP | FDIV | FMOD | UMOD | SMOD | SQUOT | SREM | SMIN | UMIN | FMIN | SMAX | UMAX | FMAX | SHL | LSHR | ASHR | AND | OR | XOR | POW | FPOW | LOGAND | LOGOR;

pConvOp: 'sext' pPrimType pSubExp 'to' pPrimType;

pSubExp: pPrimValue #ConstantSubExpression
    | ID #VarSubExp;

pPrimValue: NUMBER INTEGER_TYPE #PrimValueInteger
    | FLOAT FLOAT_TYPE #PrimValueFloat
    | ('true' | 'false') #PrimValueBool;

pPrimType: INTEGER_TYPE #PrimTypeInteger
 | FLOAT_TYPE #PrimTypeFloat
 | 'bool' #PrimTypeBool;

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
ADDNW: 'add_nw' NUMBER;
SUBNW: 'sub_nw' NUMBER;
MULNW: 'mul_nw' NUMBER;
UDIV: 'udiv' NUMBER;
UDIVUP: 'udiv_up' NUMBER;
SDIV: 'sdiv' NUMBER;
SDIVUP: 'sdiv_up' NUMBER;
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

// Comparison operators. eq_ carries a type name (eq_i64, eq_f32, eq_bool);
// the ordered comparisons carry a bit width.
CMPEQ: 'eq_' [a-z] [a-z0-9]*;
CMPSLT: 'slt' NUMBER;
CMPSLE: 'sle' NUMBER;
CMPULT: 'ult' NUMBER;
CMPULE: 'ule' NUMBER;

FLOAT   : (DIGIT+ '.' DIGIT+) ;
fragment DIGIT  : [0-9];

STRING_LITERAL: '"' ( '\\' . | ~["\\] )* '"';

OPAQUE: '*'? 'opaque';

ID : ( [a-zA-Z_+'-'*/%!<>|&^.] ) ([a-zA-Z0-9_+\-*/%!<>|&^.₀-₉])*;

LPARAM      : '(';
RPARAM      : ')';

OPEN_BRACKET: '{';
CLOSE_BRACKET: '}';

COMMA       : ',';
BUILTIN     : 'builtin';
FUNCTIONS      : 'FUNCTIONS';
RULES      : 'RULES';

WS          : [ \t\r\n]+ -> skip ;
