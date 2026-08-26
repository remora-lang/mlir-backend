#pragma once

#include "ir/core.hpp"
#include "utils.hpp"
#include <optional>
#include <string>

enum class BlackBox { MatMul, DotGeneral, RaggedDot, ArgSort };

inline BlackBox ToBlackBox(const std::string &name) {
  if (name == "matmul") {
    return BlackBox::MatMul;
  }
  if (name == "dot_general") {
    return BlackBox::DotGeneral;
  }
  if (name == "ragged_dot") {
    return BlackBox::RaggedDot;
  }
  if (name == "argsort") {
    return BlackBox::ArgSort;
  }
  Undefined();
}

// Returns the first "blackbox" found in the attributes.
inline std::optional<BlackBox> MaybeBlackBox(const Attrs &attrs) {
  for (const auto &attr : attrs.attrs) {
    if (auto *a = std::get_if<CompAttr>(&attr.v)) {
      if (a->name == "blackbox" && a->args.size() == 1) {
        if (auto *v = std::get_if<AtomAttr>(&a->args[0].v)) {
          if (v->name == "dummy")
            continue;
          return ToBlackBox(v->name);
        }
      }
    }
  }
  return std::nullopt;
}

inline bool IsBlackBox(const Attrs &attrs) {
  return MaybeBlackBox(attrs).has_value();
}
