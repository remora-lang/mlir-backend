#pragma once
// src/Futhark/IR/GPU/Op.hs with rep SOAC
#include "core.hpp"

struct SegOp;

using Name = std::string;

// Mirrors Futhark.IR.GPU.Sizes.SizeClass. Held as the raw printed name for now
// (e.g. "thread_block_size"). TODO: model faithfully as a variant, if needed
// (SizeThreshold/SizeGrid/SizeRegisters/...).
struct SizeClass {
  std::string name;
};

struct GetSize {
  Name name;
  SizeClass cls;
};

struct GetSizeMax {
  SizeClass cls;
};

struct CmpSizeLe {
  Name name;
  SizeClass cls;
  SubExp x;
};

struct CalcNumBlocks {
  SubExp w;
  Name maxNumBlocks;
  SubExp blockSize;
};

struct SizeOp {
  std::variant<GetSize, GetSizeMax, CmpSizeLe, CalcNumBlocks> v;
};

// OtherOp (SOAC) is omitted: we assume post-unstream GPU IR, which has no SOACs.
struct HostOp {
  std::variant<std::shared_ptr<SegOp>, SizeOp> v;
};
