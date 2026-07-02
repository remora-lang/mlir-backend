#pragma once
#include <stdexcept>

[[noreturn]] inline void Unreachable() {
  throw std::runtime_error("unreachable");
}
