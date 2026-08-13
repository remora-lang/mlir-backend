#pragma once
#include <format>
#include <source_location>
#include <stdexcept>

[[noreturn]] inline void
Undefined(std::source_location loc = std::source_location::current()) {
  throw std::runtime_error(
      std::format("undefined at {}:{}", loc.file_name(), loc.line()));
}
