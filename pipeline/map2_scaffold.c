// Scaffold for map2.fut_gpu: entry_main(i64 n) -> [n]i64, result[i] = n + i.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int64_t *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
} MemRef;

void _mlir_ciface_entry_main(MemRef *result, int64_t n);

int main(int argc, char **argv) {
  int64_t n = argc > 1 ? atoll(argv[1]) : 5;
  MemRef r;
  _mlir_ciface_entry_main(&r, n);
  printf("[");
  for (int64_t i = 0; i < r.sizes[0]; i++) {
    if (i) printf(", ");
    printf("%lld", (long long)r.aligned[i]);
  }
  printf("]\n");
  free(r.allocated);
  return 0;
}
