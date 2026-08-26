-- dot_general, 3-D batched case: [b][n][k] . [b][k][m] -> [b][n][m].
-- Batch axis 0; contracts lhs axis 2 with rhs axis 1.
-- rhs is a per-batch identity, so result == lhs (spec.md dot_general example).

#[noinline]
#[blackbox(dot_general)]
def dot_general [b] [n] [k] [m] [nb] [nc] (x: [b][n][k]i64) (y: [b][k][m]i64)
                (lhs_batching_dims: [nb]i64) (rhs_batching_dims: [nb]i64)
                (lhs_contracting_dims: [nc]i64) (rhs_contracting_dims: [nc]i64): [b][n][m]i64 =
  #[unsafe] ???

-- ==
-- input {
--   [[[1i64, 2i64], [3i64, 4i64]], [[5i64, 6i64], [7i64, 8i64]]]
--   [[[1i64, 0i64], [0i64, 1i64]], [[1i64, 0i64], [0i64, 1i64]]]
-- }
-- output {
--   [[[1i64, 2i64], [3i64, 4i64]], [[5i64, 6i64], [7i64, 8i64]]]
-- }
-- Batch axis 0; contract lhs axis 2 with rhs axis 1.
def main (x: [2][2][2]i64) (y: [2][2][2]i64) : [2][2][2]i64 =
  dot_general x y [0i64] [0i64] [2i64] [1i64]
