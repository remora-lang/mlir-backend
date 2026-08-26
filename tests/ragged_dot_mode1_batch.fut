-- Tests mode 1 with a batch dimension. Ragged dim is a lhs non-contracting dim.
-- Shape signature: [b][m][k] . [g][b][k][n] . [b][g] -> [b][m][n].
-- With a batch dim, group_sizes gains a leading batch axis ([b][g]).
--
-- Two batches reusing k = n = 2, g = 2. rhs group 0 is the identity and group 1
-- is upper triangular in both batches; the batches differ only in group_sizes,
-- which is why they reproduce the two no-batch cases of ragged_dot_mode1.fut.

#[noinline] #[blackbox(ragged_dot)]
def ragged_dot [b] [m] [k] [g] [n] [nb] [nc] [ng]
               (lhs: [b][m][k]i64)
               (rhs: [g][b][k][n]i64)
               (group_sizes: [b][g]i64)
               (lhs_batching_dims: [nb]i64)
               (rhs_batching_dims: [nb]i64)
               (lhs_contracting_dims: [nc]i64)
               (rhs_contracting_dims: [nc]i64)
               (lhs_ragged_dim: i64)
               (rhs_group_dims: [ng]i64) : [b][m][n]i64 =
  #[unsafe] ???

-- ==
-- input {
--   [[[1i64, 0i64], [1i64, 1i64], [2i64, 1i64], [0i64, 3i64]],
--    [[1i64, 0i64], [1i64, 1i64], [2i64, 1i64], [0i64, 3i64]]]
--   [[[[1i64, 0i64], [0i64, 1i64]], [[1i64, 0i64], [0i64, 1i64]]],
--    [[[1i64, 1i64], [0i64, 1i64]], [[1i64, 1i64], [0i64, 1i64]]]]
--   [[1i64, 3i64], [2i64, 2i64]]
-- }
-- output {
--   [[[1i64, 0i64], [1i64, 2i64], [2i64, 3i64], [0i64, 3i64]],
--    [[1i64, 0i64], [1i64, 1i64], [2i64, 3i64], [0i64, 3i64]]]
-- }
-- Batch is lhs axis 0 / rhs axis 1; ragged dim is lhs axis 1 (m);
-- contract lhs axis 2 with rhs axis 2; rhs group dim is axis 0.
def main (x: [2][4][2]i64) (y: [2][2][2][2]i64) (group_sizes: [2][2]i64) : [2][4][2]i64 =
  ragged_dot x y group_sizes [0i64] [1i64] [2i64] [2i64] 1i64 [0i64]
