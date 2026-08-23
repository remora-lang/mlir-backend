#pragma once
#include "syntax.hpp"

// Futhark.IR.SegOp.SegVirt. `SegNoVirtFull` also names the sequentialised
// dimensions; we do not model those.
enum class SegVirt { NoVirt, Virt, NoVirtFull };

// The levels also carry a KernelGrid (block count and block size) in Futhark.
// We never see one, because the IR is printed with --no-grid.
struct SegThread {
  SegVirt virt = SegVirt::NoVirt;
};

struct SegBlock {
  SegVirt virt = SegVirt::NoVirt;
};

struct SegThreadInBlock {};

struct SegSpace {
  std::string flat_id;
  std::vector<std::tuple<std::string, SubExp>> dims;
};

using SegLevel = std::variant<SegThread, SegBlock, SegThreadInBlock>;

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

struct HistOp {
  Shape shape;
  SubExp raceFactor;
  std::vector<VName> dest;
  std::vector<SubExp> neutral;
  // See the Futhark source comment about opShape.
  std::vector<SubExp> opShape;
  Lambda lambda;
};

struct SegMap {
  SegLevel lvl;
  SegSpace space;
  std::vector<Type> ret;
  KernelBody body;
};

struct SegRed {
  SegLevel lvl;
  SegSpace space;
  std::vector<Type> ret;
  KernelBody body;
  std::vector<SegBinOp> ops;
};

struct SegScan {
  SegLevel lvl;
  SegSpace space;
  std::vector<Type> ret;
  KernelBody body;
  std::vector<SegBinOp> ops;
  SegPostOp post_op;
};

struct SegHist {
  SegLevel lvl;
  SegSpace space;
  std::vector<Type> ret;
  KernelBody body;
  std::vector<HistOp> ops;
};

struct SegOp {
  std::variant<SegMap, SegRed,SegScan, SegHist> v;
};
