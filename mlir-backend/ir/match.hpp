#pragma once
#include <variant>
#include <utility>

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class V, class... Fs> decltype(auto) match(V &&v, Fs &&...fs) {
  return std::visit(overloaded{std::forward<Fs>(fs)...}, std::forward<V>(v));
}
