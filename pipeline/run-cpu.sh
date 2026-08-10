#!/usr/bin/env bash
# Compile an MLIR module (linalg-on-tensors) to a native CPU executable via a C
# scaffold, then run it. Modeled on ~/git/full-pipeline/remora2exe.
#
#   compile ./mlir-backend/tests/map.fut_gpu | ./pipeline/run-cpu.sh
#   ./pipeline/run-cpu.sh module.mlir
#
# The scaffold (default: this script's dir / scaffold.c) declares entry_main with
# its memref ABI, calls it, and prints results. Edit it per entry signature, or:
#   SCAFFOLD=path/to/scaffold.c ./pipeline/run-cpu.sh ...
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scaffold="${SCAFFOLD:-$here/scaffold.c}"
input="${1:-/dev/stdin}"

opt="$(mktemp --suffix=.mlir)"
ll="$(mktemp --suffix=.ll)"
exe="$(mktemp)"
trap 'rm -f "$opt" "$ll" "$exe"' EXIT

mlir-opt "$input" \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -buffer-deallocation-pipeline \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -finalize-memref-to-llvm \
  -convert-arith-to-llvm \
  -convert-func-to-llvm="use-bare-ptr-memref-call-conv=0" \
  -reconcile-unrealized-casts \
  -o "$opt"

mlir-translate --mlir-to-llvmir "$opt" -o "$ll"

clang "$ll" "$scaffold" -Wall -Wno-override-module -o "$exe"

"$exe"
