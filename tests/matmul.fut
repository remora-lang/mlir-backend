#[noinline]
#[blackbox(matmul)]
def matmul_f32 [n][k][m] (x: [n][k]f32) (y: [k][m]f32): [n][m]f32 =
  #[unsafe] ???

-- ==
-- input {
--   [[1f32, 2f32, 3f32], [4f32, 5f32, 6f32]]
--   [[1f32, 2f32], [3f32, 4f32], [5f32, 6f32]]
-- }
-- output {
--   [[22f32, 28f32], [49f32, 64f32]]
-- }
def main x y = matmul_f32 x y
