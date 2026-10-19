#[noinline]
#[blackbox(matmul)]
def matmul_f32 [n][k][m] (x: [n][k]f32) (y: [k][m]f32): [n][m]f32 =
  #[unsafe] ???

-- Static-shape matmul for GPU benchmarking; the fixed sizes let IREE specialize
-- tiling and codegen that dynamic shapes would preclude.
-- ==
-- random input { [10000][10000]f32 [10000][10000]f32 }
def main (x: [10000][10000]f32) (y: [10000][10000]f32) = matmul_f32 x y
