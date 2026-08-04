#pragma once
#include <utility>
#include <variant>

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class V, class... Fs> decltype(auto) match(V &&v, Fs &&...fs) {
  return std::visit(overloaded{std::forward<Fs>(fs)...}, std::forward<V>(v));
}
