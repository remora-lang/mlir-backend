// Scaffold for map3.fut_gpu:
//   entry_main(i64 rows, i64 cols, [rows][cols]i32 xss, i32 c) -> [rows][cols]i32
//   result[i][j] = xss[i][j] + c, with xss[i][j] = i*cols + j.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int32_t *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} MemRef;

void _mlir_ciface_entry_main(MemRef *result, int64_t rows, int64_t cols,
                             MemRef *xss, int32_t c);

int main(int argc, char **argv) {
  int64_t rows = argc > 1 ? atoll(argv[1]) : 3;
  int64_t cols = argc > 2 ? atoll(argv[2]) : 4;
  int32_t c = argc > 3 ? atoi(argv[3]) : 100;

  int32_t *data = malloc(rows * cols * sizeof(int32_t));
  for (int64_t i = 0; i < rows; i++)
    for (int64_t j = 0; j < cols; j++)
      data[i * cols + j] = (int32_t)(i * cols + j);

  MemRef xss = {data, data, 0, {rows, cols}, {cols, 1}};
  MemRef r;
  _mlir_ciface_entry_main(&r, rows, cols, &xss, c);

  printf("[");
  for (int64_t i = 0; i < r.sizes[0]; i++) {
    if (i) printf(", ");
    printf("[");
    for (int64_t j = 0; j < r.sizes[1]; j++) {
      if (j) printf(", ");
      printf("%d", r.aligned[r.offset + i * r.strides[0] + j * r.strides[1]]);
    }
    printf("]");
  }
  printf("]\n");

  free(data);
  free(r.allocated);
  return 0;
}
