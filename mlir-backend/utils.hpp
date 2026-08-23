#pragma once
#include <format>
#include <source_location>
#include <stdexcept>
#include <utility>
#include <variant>

// Matching on sum types.
template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};

template <typename T> inline constexpr bool is_variant_v = false;
template <typename... Ts>
inline constexpr bool is_variant_v<std::variant<Ts...>> = true;

template <typename T>
concept Variant = is_variant_v<std::remove_cvref_t<T>>;

template <typename T>
concept StructWithVariant =
    requires(T &t) { requires Variant<decltype((t.v))>; };

template <typename V, typename... Fs>
decltype(auto) match(V &&v, Fs &&...fs) {
  auto ov = overloaded{std::forward<Fs>(fs)...};
  if constexpr (Variant<V>)
    return std::visit(ov, std::forward<V>(v));
  else if constexpr (StructWithVariant<std::remove_cvref_t<V>>)
    return std::visit(ov, std::forward<V>(v).v);
  else
    static_assert(false,
                  "type error: match expects a std::variant or a struct with a "
                  "variant `.v` member");
}

// Errors.
[[noreturn]] inline void
Undefined(std::source_location loc = std::source_location::current()) {
  throw std::runtime_error(
      std::format("undefined at {}:{}", loc.file_name(), loc.line()));
}
