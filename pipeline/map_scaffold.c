// Scaffold for map.fut_gpu: entry_main(i32 n) -> [4]i32, result[i] = [1,2,3,4][i] + n.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int32_t *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
} MemRef;

void _mlir_ciface_entry_main(MemRef *result, int32_t n);

int main(int argc, char **argv) {
  int32_t n = argc > 1 ? atoi(argv[1]) : 10;
  MemRef r;
  _mlir_ciface_entry_main(&r, n);
  printf("[");
  for (int64_t i = 0; i < r.sizes[0]; i++) {
    if (i) printf(", ");
    printf("%d", r.aligned[i]);
  }
  printf("]\n");
  free(r.allocated);
  return 0;
}
