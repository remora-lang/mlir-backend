#pragma once

#include "core.hpp"
#include "error.hpp"
#include <optional>
#include <string>

enum class BlackBox { MatMul, DotGeneral };

inline BlackBox ToBlackBox(const std::string &name) {
  if (name == "matmul") {
    return BlackBox::MatMul;
  }
  if (name == "dot_general") {
    return BlackBox::DotGeneral;
  }
  Undefined();
}

// Returns the first "blackbox" found in the attributes.
inline std::optional<BlackBox> MaybeBlackBox(const Attrs &attrs) {
  for (const auto &attr : attrs.attrs) {
    if (auto *a = std::get_if<CompAttr>(&attr.v)) {
      if (a->name == "blackbox" && a->args.size() == 1) {
        if (auto *v = std::get_if<AtomAttr>(&a->args[0].v)) {
          return ToBlackBox(v->name);
        }
      }
    }
  }
  return std::nullopt;
}

inline bool IsBlackBox(const Attrs &attrs) {
  for (const auto &attr : attrs.attrs)
    if (auto *a = std::get_if<CompAttr>(&attr.v))
      if (a->name == "blackbox" && a->args.size() == 1)
        return true;
  return false;
}
