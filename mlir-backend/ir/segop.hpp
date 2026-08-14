#pragma once
#include "syntax.hpp"

struct SegThread {};

struct SegBlock {};

struct SegThreadInBlock {};

struct SegSpace {
  std::string flat_id;
  std::vector<std::tuple<std::string, SubExp>> dims;
};

struct KernelResult {
  SubExp result;
};

struct KernelBody {
  std::vector<Stm> stms;
  std::vector<KernelResult> result;
};

struct SegBinOp {
  Commutativity comm = Commutativity::Noncommutative;
  Lambda lambda;
  std::vector<SubExp> neutral;
  Shape shape;
};

struct SegPostOp {
  Lambda lambda;
};

struct SegMap {
  std::variant<SegThread, SegBlock, SegThreadInBlock> lvl;
  SegSpace space;
  std::vector<Type> ret;
  KernelBody body;
};

struct SegRed {
  std::variant<SegThread, SegBlock, SegThreadInBlock> lvl;
  SegSpace space;
  std::vector<Type> ret;
  KernelBody body;
  std::vector<SegBinOp> ops;
};

struct SegScan {
  std::variant<SegThread, SegBlock, SegThreadInBlock> lvl;
  SegSpace space;
  std::vector<Type> ret;
  KernelBody body;
  std::vector<SegBinOp> ops;
  SegPostOp post_op;
};

struct SegOp {
  std::variant<SegMap, SegRed,SegScan> v;
};
