// Scaffold driver for run-cpu.sh: declares entry_main with its memref ABI,
// calls it, and prints the result. Edit per entry signature.
//
// The compiler lowers `(%arg0: i32) -> tensor<4xi32>` to an LLVM function that
// returns the memref descriptor BY VALUE and takes the scalar arg normally:
//   { ptr, ptr, i64, [1 x i64], [1 x i64] } @entry_main(i32)
#include <stdio.h>
#include <stdint.h>

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRef1DI32;

extern MemRef1DI32 entry_main(int32_t arg0);

int main(void) {
  MemRef1DI32 r = entry_main(10);
  printf("[");
  for (int64_t i = 0; i < r.sizes[0]; i++) {
    if (i != 0) printf(", ");
    printf("%d", r.aligned[r.offset + i]);
  }
  printf("]\n");
  return 0;
}
