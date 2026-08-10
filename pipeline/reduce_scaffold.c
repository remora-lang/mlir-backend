// Scaffold for reduce.fut_gpu: entry_main(i64 n, [n]i32 xs) -> i32, result = sum xs.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int32_t *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
} MemRef;

int32_t _mlir_ciface_entry_main(int64_t n, MemRef *xs);

int main(int argc, char **argv) {
  int64_t n = argc > 1 ? atoll(argv[1]) : 5;

  int32_t *data = malloc(n * sizeof(int32_t));
  for (int64_t i = 0; i < n; i++)
    data[i] = (int32_t)(i + 1);

  MemRef xs = {data, data, 0, {n}, {1}};
  int32_t r = _mlir_ciface_entry_main(n, &xs);
  printf("%d\n", r);

  free(data);
  return 0;
}
