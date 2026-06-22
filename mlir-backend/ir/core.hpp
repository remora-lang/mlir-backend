#pragma once
#include "primitive.hpp"

struct VName {
  std::string name;
  int64_t tag = 0;
};

enum Commutativity {
  Noncommutative,
  Commutative,
};

enum class Uniqueness { Nonunique, Unique };

struct NoUniqueness {};

template <typename D> struct ShapeBase {
  std::vector<D> dims;
};

struct SubExp;

template <typename D> struct DimFix {
  D i;
};

template <typename D> struct DimSlice {
  D start;
  D length;
  D stride;
};

template <typename D> struct DimIndex {
  std::variant<DimFix<D>, DimSlice<D>> v;
};

template <typename D> struct Slice {
  std::vector<DimIndex<D>> dims;
};

template <typename D> struct FlatDimIndex {
  D n;
  D s;
};

template <typename D> struct FlatSlice {
  D offset;
  std::vector<FlatDimIndex<D>> dims;
};

using Shape = ShapeBase<SubExp>;

template <typename A> struct Ext {
  std::variant<int, std::shared_ptr<A>> v;
};

using ExtSize = Ext<SubExp>;
using ExtShape = ShapeBase<ExtSize>;

struct Type;

// TODO: Implement
struct Space {};

template <typename ShapeT, typename UT> struct TypePrim {
  PrimType t;
};

template <typename ShapeT, typename UT> struct TypeAcc {
  PrimType t;
  Shape ispace;
  std::vector<Type> ts;
  UT u;
};

template <typename ShapeT, typename UT> struct TypeArray {
  PrimType elem;
  ShapeT shape;
  UT u;
};

template <typename ShapeT, typename UT> struct TypeMem {
  Space space;
};

template <typename ShapeT, typename UT> struct TypeBase {
  std::variant<TypePrim<ShapeT, UT>, TypeAcc<ShapeT, UT>, TypeArray<ShapeT, UT>, TypeMem<ShapeT, UT>> v;
};

struct Type {
  TypeBase<Shape, NoUniqueness> t;

  static Type CreatePrim(PrimType primType) {
    auto arr = TypePrim<Shape, NoUniqueness>{primType};
    auto typeBase = TypeBase<Shape, NoUniqueness>{arr};
    return Type{typeBase};
  }

  static Type CreateArr(PrimType primType, Shape shape) {
    auto arr = TypeArray<Shape, NoUniqueness>{primType, shape, NoUniqueness{}};
    auto typeBase = TypeBase<Shape, NoUniqueness>{arr};
    return Type{typeBase};
  }
};

struct ExtType {
  TypeBase<ExtShape, NoUniqueness> t;
};

struct DeclType {
  TypeBase<Shape, Uniqueness> t;
};

struct DeclExtType {
  TypeBase<ExtShape, Uniqueness> t;
};

enum class Diet {
  Consume,
  Observe,
  ObservePrim,
};

struct Certs {
  std::vector<VName> vnames;
};

struct ConstantSubExp {
  PrimValue v;
};

struct VarSubExp {
  VName v;
};

struct SubExp {
  std::variant<ConstantSubExp, VarSubExp> v;

  int64_t GetIntValue() {
    auto constExp = std::get<ConstantSubExp>(v);
    auto iv = std::get<IntValue>(constExp.v.v);
    return GetValue(iv);
  }
};

template <typename Dec> struct PatElem {
  VName name;
  Dec dec;
};

template <typename Dec> struct Pat {
  std::vector<PatElem<Dec>> elems;
};

struct AtomAttr {
  std::string name;
};

struct IntAttr {
  int64_t i;
};

struct CompAttr {
  std::string name;
  std::vector<struct Attr> args;
};

struct Attr {
  std::variant<AtomAttr, IntAttr, CompAttr> v;
};

struct Attrs {
  std::vector<Attr> attrs;
};

template <typename Dec> struct Param {
  Attrs attrs;
  std::string name;
  Dec dec;
};

struct PrimExp;

struct LeafExp {
  std::string v;
  PrimType t;
};

struct ValueExp {
  PrimValue v;
};

struct BinOpExp {
  BinOp op;
  std::shared_ptr<PrimExp> x;
  std::shared_ptr<PrimExp> y;
};

struct CmpOpExp {
  CmpOp op;
  std::shared_ptr<PrimExp> x;
  std::shared_ptr<PrimExp> y;
};

struct UnOpExp {
  UnOp op;
  std::shared_ptr<PrimExp> x;
};

struct ConvOpExp {
  ConvOp op;
  std::shared_ptr<PrimExp> x;
};

struct FunExp {
  std::string fname;
  std::vector<PrimExp> args;
  PrimType t;
};

struct PrimExp {
  std::variant<LeafExp, ValueExp, BinOpExp, CmpOpExp, UnOpExp, ConvOpExp, FunExp> v;
};