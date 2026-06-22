#pragma once

enum IntType {
  Int8 = 8,
  Int16 = 16,
  Int32 = 32,
  Int64 = 64,
};

uint32_t GetWidth(IntType t) {
  switch (t) {
  case IntType::Int8:
    return 8;
  case IntType::Int16:
    return 16;
  case IntType::Int32:
    return 32;
  default:
    return 64;
  }
}

enum FloatType {
  Float16 = 16,
  Float32 = 32,
  Float64 = 64,
};

uint32_t GetWidth(FloatType t) {
  switch (t) {
  case FloatType::Float16:
    return 16;
  case FloatType::Float32:
    return 32;
  default:
    return 64;
  }
}

struct PrimTypeInt {
  IntType t;
};

struct PrimTypeFloat {
  FloatType t;
};

struct PrimTypeBool {};
struct PrimTypeUnit {};

struct PrimType {
  std::variant<PrimTypeInt, PrimTypeFloat, PrimTypeBool, PrimTypeUnit> v;

  static PrimType Int8() { return Int(8); }

  static PrimType Int16() { return Int(16); }

  static PrimType Int32() { return Int(32); }

  static PrimType Int64() { return Int(64); }

  static PrimType Int(uint32_t v) {
    IntType t;
    switch (v) {
    case 8:
      t = IntType::Int8;
      break;
    case 16:
      t = IntType::Int16;
      break;
    case 32:
      t = IntType::Int32;
      break;
    case 64:
      t = IntType::Int64;
      break;
    }

    return {PrimTypeInt{t}};
  }

  static PrimType Float16() { return Float(16); }

  static PrimType Float32() { return Float(32); }

  static PrimType Float64() { return Float(64); }

  static PrimType Float(uint32_t v) {
    FloatType t;
    switch (v) {
    case 16:
      t = FloatType::Float16;
      break;
    case 32:
      t = FloatType::Float32;
      break;
    case 64:
      t = FloatType::Float16;
      break;
    }

    return {PrimTypeFloat{t}};
  }
};

struct Int8Value {
  int8_t v;
};

struct Int16Value {
  int16_t v;
};

struct Int32Value {
  int32_t v;
};

struct Int64Value {
  int64_t v;
};

struct IntValue {
  std::variant<Int8Value, Int16Value, Int32Value, Int64Value> v;

  static IntValue CreateInt8(int8_t value) { return {Int8Value{value}}; }
  static IntValue CreateInt16(int16_t value) { return {Int16Value{value}}; }
  static IntValue CreateInt32(int32_t value) { return {Int32Value{value}}; }
  static IntValue CreateInt64(int64_t value) { return {Int64Value{value}}; }
};

uint64_t GetValue(IntValue v) {
  if (auto *val = std::get_if<Int8Value>(&v.v)) {
    return val->v;
  }

  if (auto *val = std::get_if<Int16Value>(&v.v)) {
    return val->v;
  }

  if (auto *val = std::get_if<Int32Value>(&v.v)) {
    return val->v;
  }

  if (auto *val = std::get_if<Int64Value>(&v.v)) {
    return val->v;
  }

  throw std::runtime_error("unreachable");
}

uint64_t GetWidth(IntValue v) {
  if (auto *val = std::get_if<Int8Value>(&v.v)) {
    return 8;
  }

  if (auto *val = std::get_if<Int16Value>(&v.v)) {
    return 16;
  }

  if (auto *val = std::get_if<Int32Value>(&v.v)) {
    return 32;
  }

  if (auto *val = std::get_if<Int64Value>(&v.v)) {
    return 64;
  }

  throw std::runtime_error("unreachable");
}

struct Float16Value {
  uint16_t bits;
};

struct Float32Value {
  float v;
};

struct Float64Value {
  double v;
};

struct FloatValue {
  std::variant<Float16Value, Float32Value, Float64Value> v;
};

double GetValue(FloatValue v) {
  // TODO: Cast Float16 properly
  if (auto *val = std::get_if<Float16Value>(&v.v)) {
    return (double)val->bits;
  }

  if (auto *val = std::get_if<Float32Value>(&v.v)) {
    return val->v;
  }

  if (auto *val = std::get_if<Float64Value>(&v.v)) {
    return val->v;
  }

  throw std::runtime_error("unreachable");
}

uint64_t GetWidth(FloatValue v) {
  if (auto *val = std::get_if<Float16Value>(&v.v)) {
    return 16;
  }

  if (auto *val = std::get_if<Float32Value>(&v.v)) {
    return 32;
  }

  if (auto *val = std::get_if<Float64Value>(&v.v)) {
    return 64;
  }

  throw std::runtime_error("unreachable");
}

struct BoolValue {
  bool v;
};

struct UnitValue {};

struct PrimValue {
  std::variant<IntValue, FloatValue, BoolValue, UnitValue> v;

  static PrimValue CreateInt(int64_t value, uint64_t width) {
    if (width == 8)
      return {IntValue::CreateInt8(value)};
    if (width == 16)
      return {IntValue::CreateInt16(value)};
    if (width == 32)
      return {IntValue::CreateInt32(value)};
    return {IntValue::CreateInt64(value)};
  }
};

enum Overflow {
  OverflowWrap,
  OverflowUndef,
};

enum Safety {
  Unsafe,
  Safe,
};

struct UnOpNeg {
  PrimType t;
};

struct UnOpComplement {
  IntType t;
};

struct UnOpAbs {
  IntType t;
};

struct UnOpFAbs {
  FloatType t;
};

struct UnOpSSignum {
  IntType t;
};

struct UnOpUSignum {
  IntType t;
};

struct UnOpFSignum {
  FloatType t;
};

struct UnOp {
  std::variant<UnOpNeg, UnOpComplement, UnOpAbs, UnOpFAbs, UnOpSSignum, UnOpUSignum, UnOpFSignum> v;
};

struct BinOpAdd {
  IntType t;
  Overflow ovf;
};

struct BinOpFAdd {
  FloatType t;
};

struct BinOpSub {
  IntType t;
  Overflow ovf;
};

struct BinOpFSub {
  FloatType t;
};

struct BinOpMul {
  IntType t;
  Overflow ovf;
};

struct BinOpFMul {
  FloatType t;
};

struct BinOpUDiv {
  IntType t;
  Safety s;
};

struct BinOpSDiv {
  IntType t;
  Safety s;
};

struct BinOpSDivUp {
  IntType t;
  Safety s;
};

struct BinOpFDiv {
  FloatType t;
};

struct BinOpFMod {
  FloatType t;
};

struct BinOpUMod {
  IntType t;
  Safety s;
};

struct BinOpSMod {
  IntType t;
  Safety s;
};

struct BinOpSQuot {
  IntType t;
  Safety s;
};

struct BinOpSRem {
  IntType t;
  Safety s;
};

struct BinOpSMin {
  IntType t;
};

struct BinOpUMin {
  IntType t;
};

struct BinOpFMin {
  FloatType t;
};

struct BinOpSMax {
  IntType t;
};

struct BinOpUMax {
  IntType t;
};

struct BinOpFMax {
  FloatType t;
};

struct BinOpShl {
  IntType t;
};

struct BinOpLShr {
  IntType t;
};

struct BinOpAShr {
  IntType t;
};

struct BinOpAnd {
  IntType t;
};

struct BinOpOr {
  IntType t;
};

struct BinOpXor {
  IntType t;
};

struct BinOpPow {
  IntType t;
};

struct BinOpFPow {
  IntType t;
};

struct BinOpLogAnd {};

struct BinOpLogOr {};

// TODO: Implement rest of operators
struct BinOp {
  std::variant<BinOpAdd, BinOpFAdd, BinOpSub, BinOpFSub, BinOpMul, BinOpFMul, BinOpUDiv, BinOpUDiv, BinOpSDiv,
               BinOpSDivUp, BinOpFDiv, BinOpFMod, BinOpUMod, BinOpSMod, BinOpSQuot, BinOpSRem, BinOpSMin, BinOpUMin,
               BinOpFMin, BinOpSMax, BinOpUMax, BinOpFMax, BinOpShl, BinOpLShr, BinOpAShr, BinOpAnd, BinOpOr, BinOpXor,
               BinOpPow, BinOpFPow, BinOpLogAnd, BinOpLogOr>
      v;
};

struct CmpOpEq {
  PrimType t;
};

struct CmpOpUlt {
  IntType t;
};

struct CmpOpUle {
  IntType t;
};

struct CmpOpSlt {
  IntType t;
};

struct CmpOpSle {
  IntType t;
};

struct CmpOpFlt {
  FloatType t;
};

struct CmpOpFle {
  FloatType t;
};

struct CmpOpLLt {};
struct CmpOpLLe {};

struct CmpOp {
  std::variant<CmpOpEq, CmpOpUlt, CmpOpUle, CmpOpSlt, CmpOpSle, CmpOpFlt, CmpOpFle, CmpOpLLt, CmpOpLLe> v;
};

struct ConvOpZExt {
  IntType from;
  IntType to;
};

struct ConvOpSExt {
  IntType from;
  IntType to;
};

// TODO: Rest of ConvOp operators
struct ConvOp {
  std::variant<ConvOpZExt, ConvOpSExt> v;
};