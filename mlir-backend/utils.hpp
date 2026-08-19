#pragma once
#include <format>
#include <source_location>
#include <stdexcept>
#include <utility>
#include <variant>

// Matching on union types.
template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class V, class... Fs> decltype(auto) match(V &&v, Fs &&...fs) {
  return std::visit(overloaded{std::forward<Fs>(fs)...}, std::forward<V>(v));
}

// Errors.
[[noreturn]] inline void
Undefined(std::source_location loc = std::source_location::current()) {
  throw std::runtime_error(
      std::format("undefined at {}:{}", loc.file_name(), loc.line()));
}
