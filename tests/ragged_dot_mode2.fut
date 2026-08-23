-- Tests mode 2. Ragged dimension is the lhs/rhs contracting dim.
-- Shape signature (no batch): [m][k] . [k][n] . [g] -> [g][m][n].
-- group_sizes partitions the contracting dim k into g contiguous groups;
-- unlike a normal dot the groups are NOT summed together, so the result
-- gains a leading group axis: output[i] contracts only k-slice i.
--
-- Here m = n = 2, k = 4, g = 2 with an even split [2, 2]. rhs is stacked
-- identities, so each group just copies the matching 2 columns of lhs.

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
--   [[1i64, 2i64, 3i64, 4i64], [5i64, 6i64, 7i64, 8i64]]
--   [[1i64, 0i64], [0i64, 1i64], [1i64, 0i64], [0i64, 1i64]]
--   [2i64, 2i64]
-- }
-- output {
--   [[[1i64, 2i64], [5i64, 6i64]], [[3i64, 4i64], [7i64, 8i64]]]
-- }
-- Ragged dim is the contracting dim: lhs axis 1, rhs axis 0. No batch dims,
-- no rhs group dim (mode 2 adds the group axis to the result instead).
def main (x: [2][4]i64) (y: [4][2]i64) (group_sizes: [2]i64) : [2][2][2]i64 =
  ragged_dot x y group_sizes [] [] [1i64] [0i64] 1i64 []
