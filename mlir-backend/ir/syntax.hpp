#pragma once

#include "core.hpp"
#include "gpu.hpp"

enum class Rep {
  SOACS,
  Seq,
  GPU,
  MC,
  SeqMem,
  GPUMem,
  MCMem,
};

struct Body;
struct Lambda;

struct LetDec {
  Type v;
};

struct FParamInfo {
  // DeclType v;
  Type v;
};

struct LParamInfo {
  Type v;
};

struct RetType {
  Type v;
  // DeclExtType v;
};

struct BranchType {
  Type v;
  // ExtType v;
};

using FParam = Param<FParamInfo>;
using LParam = Param<LParamInfo>;

template <typename D> struct DimSplice {
  int i;
  int k;
  ShapeBase<D> shape;
};

template <typename D> struct NewShape {
  std::vector<DimSplice<D>> splices;
  ShapeBase<D> newShape;
};

struct BasicOpSubExp {
  SubExp subExp;
};

// TODO:
struct BasicOpOpaque {};

struct BasicOpArrayLit {
  std::vector<SubExp> values;
  Type t;
};

struct BasicOpArrayVal {
  std::vector<PrimValue> values;
  PrimType type;
};

struct BasicOpUnOp {
  UnOp op;
  SubExp subExp;
};

struct BasicOpBinOp {
  BinOp op;
  SubExp op0;
  SubExp op1;
};

struct BasicOpCmpOp {
  CmpOp op;
  SubExp op0;
  SubExp op1;
};

struct BasicOpConvOp {
  ConvOp op;
  SubExp op0;
};

// TODO
struct BasicOpAssert {};

struct BasicOpIndex {
  VName vName;
  Slice<SubExp> slice;
};

struct BasicOpUpdate {
  Safety safety;
  VName vName;
  Slice<SubExp> slice;
  SubExp val;
};

// `arr[offset; n1:s1, n2:s2]`. Distinct from BasicOpIndex, which is the
// ordinary `arr[i, j]` and `arr[i :+ n * s]`.
struct BasicOpFlatIndex {
  VName vName;
  FlatSlice<SubExp> slice;
};

struct BasicOpFlatUpdate {
  VName vName;
  FlatSlice<SubExp> slice;
  VName val;
};

struct BasicOpConcat {
  int dim;
  std::vector<VName> arrs;
  SubExp total;
};

struct BasicOpManifest {
  VName arr;
  std::vector<int> perm;
};

struct BasicOpIota {
  SubExp n;
  SubExp x;
  SubExp s;
  IntType t;
};

struct BasicOpReplicate {
  Shape shape;
  SubExp val;
};

struct BasicOpScratch {
  PrimType type;
  std::vector<SubExp> dims;
};

struct BasicOpReshape {
  // VName arr;
  // NewShape<SubExp> shape;

  SubExp op0;
  int dimBegin;
  int dimEnd;
  Shape reshapedDim;
  Shape remainder;
};

struct BasicOpRearrange {
  VName arr;
  std::vector<int64_t> perm;
};

struct BasicOpUpdateAcc {
  Safety safety;
  VName acc;
  std::vector<SubExp> indices;
  std::vector<SubExp> values;
};

struct BasicOp {
  std::variant<BasicOpSubExp, BasicOpOpaque, BasicOpArrayLit, BasicOpArrayVal,
               BasicOpUnOp, BasicOpBinOp, BasicOpCmpOp, BasicOpConvOp,
               BasicOpAssert, BasicOpIndex, BasicOpUpdate, BasicOpFlatIndex,
               BasicOpFlatUpdate, BasicOpConcat, BasicOpManifest, BasicOpIota,
               BasicOpReplicate, BasicOpScratch, BasicOpReshape,
               BasicOpRearrange, BasicOpUpdateAcc>
      v;
};

struct SubExpRes {
  Certs certs;
  SubExp subExp;
};

struct RetAls {
  std::vector<int> paramAls;
  std::vector<int> otherAls;
};

// TODO
struct StmAux {};

struct ExpBasicOp {
  BasicOp op;
};

struct ExpApply {
  std::string fname;
  std::vector<std::pair<SubExp, Diet>> args;
  std::vector<std::pair<RetType, RetAls>> retTypes;
  Safety safety;
};

struct ExpSubExp {
  SubExp subExp;
};

struct ExpHostOp {
  HostOp op;
};

struct WithAccInput {
  Shape ispace;
  std::vector<VName> arrays;
  std::optional<std::pair<std::shared_ptr<Lambda>, std::vector<SubExp>>> op;
};

struct ExpWithAcc {
  std::vector<WithAccInput> inputs;
  std::shared_ptr<Lambda> lambda;
};
// Futhark.IR.Syntax.MatchSort, printed between `if` and the condition:
// nothing, `<fallback>` or `<equiv>`.
enum class MatchSort { Normal, Fallback, Equiv };

// NOTE: Futhark has a more general "Match".
struct ExpIf {
  SubExp cond;
  std::shared_ptr<Body> then_body;
  std::shared_ptr<Body> else_body;
  std::vector<Type> retType;
  MatchSort sort = MatchSort::Normal;
};

// Futhark.IR.Syntax.LoopForm.
struct ForLoop {
  VName i;
  IntType t;
  SubExp bound;
};

struct WhileLoop {
  VName cond;
};

struct LoopForm {
  std::variant<ForLoop, WhileLoop> v;
};

struct ExpLoop {
  // Each merge parameter paired with its initial value.
  std::vector<std::pair<FParam, SubExp>> merge;
  LoopForm form;
  std::shared_ptr<Body> body;
};

struct Exp {
  std::variant<ExpBasicOp, ExpApply, ExpHostOp, ExpSubExp, ExpIf, ExpLoop, ExpWithAcc> v;
};

struct Stm {
  Pat<LetDec> pat;
  StmAux aux;
  Exp exp;
};

// using Result = std::vector<SubExpRes>;
using Result = std::vector<SubExp>;

struct Body {
  std::vector<Stm> stms;
  Result result;
};

struct Lambda {
  std::vector<LParam> params;
  std::vector<Type> ret;
  Body body;
};

struct EntryParam {
  std::string name;
  Uniqueness u;
};

struct EntryResult {
  std::string name;
  Uniqueness u;
};

struct EntryPoint {
  std::string name;
  std::vector<EntryParam> params;
  std::vector<EntryResult> results;
};

struct FunDef {
  std::optional<EntryPoint> entry;
  Attrs attrs;
  std::string name;
  std::vector<std::pair<RetType, RetAls>> retType;
  std::vector<FParam> params;
  Body body;

  bool operator==(const FunDef &other) const { return name == other.name; }
};

struct FunDefHasher {
  size_t operator()(const FunDef &fun) const {
    return std::hash<std::string>{}(fun.name);
  }
};

struct Prog {
  Rep rep;
  // Top-level constants that are in scope for all functions.
  std::vector<Stm> consts;
  // Top-level functions. Unlike in the Futhark compiler these are not in scope
  // in the definitions of the constants.
  std::vector<FunDef> funs;
};
