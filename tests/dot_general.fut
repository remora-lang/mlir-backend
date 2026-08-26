-- dot_general, 2-D case: [n][k] . [k][m] -> [n][m].
-- Contracts lhs axis 1 with rhs axis 0; no batch dimensions.

#[noinline]
#[blackbox(dot_general)]
def dot_general [n] [k] [m] [nb] [nc] (x: [n][k]f32) (y: [k][m]f32)
                (lhs_batching_dims: [nb]i64) (rhs_batching_dims: [nb]i64)
                (lhs_contracting_dims: [nc]i64) (rhs_contracting_dims: [nc]i64): [n][m]f32 =
  #[unsafe] ???

-- ==
-- input {
--   [[1f32, 2f32, 3f32], [4f32, 5f32, 6f32]]
--   [[1f32, 2f32], [3f32, 4f32], [5f32, 6f32]]
-- }
-- output {
--   [[22f32, 28f32], [49f32, 64f32]]
-- }
-- ==
-- input {
--   [[1f32, 2f32, 3f32], [4f32, 5f32, 6f32]]
--   [[1f32, 0f32], [0f32, 1f32], [1f32, 1f32]]
-- }
-- output {
--   [[4f32, 5f32], [10f32, 11f32]]
-- }
-- No batch dims; contract lhs axis 1 with rhs axis 0.
def main (x: [2][3]f32) (y: [3][2]f32) : [2][2]f32 =
  dot_general x y [][0i64] [] [1i64] [0i64]
