-- Tests mode 3. Ragged dimension is a lhs/rhs batch dim.
-- Shape signature: [b][m][k] . [b][k][n] . [g] -> [b][m][n].
-- group_sizes partitions the batch dim b into g contiguous groups, but the
-- groups are independent so the result is just a batched matmul: group_sizes
-- does not affect the output at all.
--
-- Here b = m = k = n = 2. rhs[0] is the identity and rhs[1] swaps columns.
-- The two test vectors use different group_sizes ([1,1] then [2,0], an empty
-- second group) to show the output is identical either way.

#[noinline]
#[blackbox(ragged_dot)]
def ragged_dot 'a 'b 'c [g] [n] [m] [s]
                (x: a) (y: b) (group_sizes: [g]i64)
                (lhs_batching_dims: [n]i64) (rhs_batching_dims: [n]i64)
                (lhs_contracting_dims: [m]i64) (rhs_contracting_dims: [m]i64)
                (lhs_ragged_dim: i64) (rhs_group_dims: [s]i64): c =
  #[unsafe] ???

-- ==
-- input {
--   [[[1i64, 2i64], [3i64, 4i64]], [[5i64, 6i64], [7i64, 8i64]]]
--   [[[1i64, 0i64], [0i64, 1i64]], [[0i64, 1i64], [1i64, 0i64]]]
--   [1i64, 1i64]
-- }
-- output {
--   [[[1i64, 2i64], [3i64, 4i64]], [[6i64, 5i64], [8i64, 7i64]]]
-- }
-- ==
-- input {
--   [[[1i64, 2i64], [3i64, 4i64]], [[5i64, 6i64], [7i64, 8i64]]]
--   [[[1i64, 0i64], [0i64, 1i64]], [[0i64, 1i64], [1i64, 0i64]]]
--   [2i64, 0i64]
-- }
-- output {
--   [[[1i64, 2i64], [3i64, 4i64]], [[6i64, 5i64], [8i64, 7i64]]]
-- }
-- Ragged dim is the batch dim: lhs axis 0. Batch axis 0; contract lhs axis 2
-- with rhs axis 1; no rhs group dim.
def main (x: [2][2][2]i64) (y: [2][2][2]i64) (group_sizes: [2]i64) : [2][2][2]i64 =
  ragged_dot x y group_sizes [0i64] [0i64] [2i64] [1i64] 0i64 []
