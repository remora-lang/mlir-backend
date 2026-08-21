-- Core irregular (ragged) matmul from dropless MoE.
--
-- Reference: MegaBlocks, Trevor Gale, arXiv:2211.15841 (Figure 1).
--
-- Multiplies each contiguous chunk of rows by its own weight matrix:
--   out[i] = xs[i] @ W[e]   for every row i in chunk e.
-- This is the kernel underneath dmoe once tokens are already grouped by
-- expert; routing, grouping, activation and scaling are all stripped out.

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

def ragged_matmul [m] [E] [d1] [d2]
                  (xs: [m][d1]f32)
                  (W: [E][d1][d2]f32)
                  (group_sizes: [E]i64) : [m][d2]f32 =
  ragged_dot xs W group_sizes [] [] [1i64] [1i64] 0i64 [0i64]

-- Testing.
--
-- Two groups: row 0 uses the identity, rows 1-2 use 2*identity.
-- ==
-- entry: ragged_matmul_test
-- input {
--   [[1f32, 2f32], [3f32, 4f32], [5f32, 6f32]]
--   [[[1f32, 0f32], [0f32, 1f32]],
--    [[2f32, 0f32], [0f32, 2f32]]]
--   [1i64, 2i64]
-- }
-- output {
--   [[1f32, 2f32], [6f32, 8f32], [10f32, 12f32]]
-- }
entry main (xs: [3][2]f32)
           (W: [2][2][2]f32)
           (group_sizes: [2]i64) : [3][2]f32 =
  ragged_matmul xs W group_sizes
