#pragma once
#include "mlir/IR/Value.h"

#include <vector>

inline void Print(std::string msg) {
  llvm::errs() << msg;
}

inline void PrintValue(mlir::Value value) {
  mlir::OpPrintingFlags flags;
  flags.assumeVerified();
  value.print(llvm::errs(), flags);
  llvm::errs() << "\n";
}

template <typename T> inline void PrintValue(const T &value) {
  llvm::errs() << value << "\n";
}

template <typename T> inline void PrintValue(const std::vector<T> &values) {
  for (const auto &value : values)
    PrintValue(value);
}

template <typename T> inline void PrintValue(const mlir::SmallVector<T> &values) {
  for (const auto &value : values)
    PrintValue(value);
}
