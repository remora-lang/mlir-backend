#pragma once
// src/Futhark/IR/GPU/Op.hs with rep SOAC
#include "core.hpp"
#include <memory>
#include <optional>

struct SegOp;
struct Body;

using Name = std::string;

// Mirrors Futhark.IR.GPU.Sizes.SizeClass.

// Printed as `threshold(<def>, <path>)`, e.g. `threshold(32, )`. The path is
// the sequence of threshold comparisons that led here; it is empty in all IR
// we have seen, so we keep it as the raw printed text.
struct SizeThreshold {
  std::optional<int64_t> def;
  std::string path;
};

struct SizeGrid {};
struct SizeBlockSize {};
struct SizeTile {};
struct SizeRegTile {};
struct SizeLocalMemory {};

// A size that is not one of the above; we keep whatever name was printed.
struct SizeBespoke {
  Name name;
};

struct SizeClass {
  std::variant<SizeThreshold, SizeGrid, SizeBlockSize, SizeTile, SizeRegTile,
               SizeLocalMemory, SizeBespoke>
      v;
};

// The names Futhark prints for the classes that carry no payload.
inline SizeClass SizeClassFromName(const Name &name) {
  if (name == "grid_size")
    return {SizeGrid{}};
  if (name == "thread_block_size")
    return {SizeBlockSize{}};
  if (name == "tile_size")
    return {SizeTile{}};
  if (name == "reg_tile_size")
    return {SizeRegTile{}};
  if (name == "shared_memory" || name == "local_memory")
    return {SizeLocalMemory{}};
  return {SizeBespoke{name}};
}

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

// A body evaluated sequentially on a single device thread.
struct GPUBody {
  std::vector<Type> retType;
  std::shared_ptr<Body> body;
};

// OtherOp (SOAC) is omitted: we assume post-unstream GPU IR, which has no
// SOACs.
struct HostOp {
  std::variant<std::shared_ptr<SegOp>, SizeOp, GPUBody> v;
};
