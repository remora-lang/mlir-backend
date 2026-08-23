-- Tests mode 1. Ragged dimension is a lhs non-contracting dim.
-- Shape signature (no batch): [m][k] . [g][k][n] . [g] -> [m][n].
-- group_sizes partitions the m rows of lhs into g contiguous groups;
-- rows in group i are multiplied by rhs[i].
--
-- Here k = n = 2 and g = 2. rhs[0] is the identity and rhs[1] is upper
-- triangular, so a row's group selects which matrix transforms it.

#[noinline] #[blackbox(ragged_dot)]
def ragged_dot 'a 'b 'c [g] [n] [m] [s]
               (lhs: a)
               (rhs: b)
               (group_sizes: [g]i64)
               (lhs_batching_dims: [n]i64)
               (rhs_batching_dims: [n]i64)
               (lhs_contracting_dims: [m]i64)
               (rhs_contracting_dims: [m]i64)
               (lhs_ragged_dim: i64)
               (rhs_group_dims: [s]i64) : c =
  #[unsafe] ???

-- ==
-- input {
--   [[1i64, 0i64], [1i64, 1i64], [2i64, 1i64], [0i64, 3i64]]
--   [[[1i64, 0i64], [0i64, 1i64]], [[1i64, 1i64], [0i64, 1i64]]]
--   [1i64, 3i64]
-- }
-- output {
--   [[1i64, 0i64], [1i64, 2i64], [2i64, 3i64], [0i64, 3i64]]
-- }
-- ==
-- input {
--   [[1i64, 0i64], [1i64, 1i64], [2i64, 1i64], [0i64, 3i64]]
--   [[[1i64, 0i64], [0i64, 1i64]], [[1i64, 1i64], [0i64, 1i64]]]
--   [2i64, 2i64]
-- }
-- output {
--   [[1i64, 0i64], [1i64, 1i64], [2i64, 3i64], [0i64, 3i64]]
-- }
-- Ragged dim is lhs axis 0 (m); contract lhs axis 1 with rhs axis 1;
-- rhs group dim is axis 0. No batch dims.
def main (x: [4][2]i64) (y: [2][2][2]i64) (group_sizes: [2]i64) : [4][2]i64 =
  ragged_dot x y group_sizes [] [] [1i64] [1i64] 0i64 [0i64]
