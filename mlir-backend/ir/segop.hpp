#pragma once
#include "syntax.hpp"

struct SegThread {};

struct SegBlock {};

struct SegThreadInBlock {};

struct SegSpace {
  std::string flat_id;
  std::vector<std::tuple<std::string,SubExp>> dims;
};

struct KernelResult {
  SubExp result;
};

struct KernelBody {
  std::vector<Stm> stms;
  std::vector<KernelResult> result;
};

struct SegMap {
  std::variant<SegThread, SegBlock, SegThreadInBlock> lvl;
  SegSpace space;
  std::vector<Type> ret;
  KernelBody body;
};

struct SegOp {
  std::variant<SegMap> v;
};
