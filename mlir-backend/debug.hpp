#pragma once
#include "mlir/IR/Value.h"

#include <mlir/IR/OpDefinition.h>
#include <type_traits>
#include <vector>

inline void Print(std::string msg) { llvm::errs() << msg; }

inline void PrintValue(mlir::Value value) {
  mlir::OpPrintingFlags flags;
  flags.assumeVerified();
  value.print(llvm::errs(), flags);
  llvm::errs() << "\n";
}

template <typename T>
  requires std::is_base_of_v<mlir::OpState, T>
inline void PrintValue(T &value) {
  mlir::OpPrintingFlags flags;
  flags.assumeVerified();
  auto op = value.getOperation();
  op->print(llvm::errs(), flags);
  llvm::errs() << "\n";
}

template <typename T> inline void PrintValue(const T &value) {
  llvm::errs() << value << "\n";
}

template <typename T> inline void PrintValue(const std::vector<T> &values) {
  for (const auto &value : values)
    PrintValue(value);
}

template <typename T>
inline void PrintValue(const mlir::SmallVector<T> &values) {
  for (const auto &value : values)
    PrintValue(value);
}
