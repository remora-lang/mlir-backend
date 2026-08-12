-- Core irregular (ragged) matmul from dropless MoE.
--
-- Reference: MegaBlocks, Trevor Gale, arXiv:2211.15841 (Figure 1).
--
-- Multiplies each contiguous chunk of rows by its own weight matrix:
--   out[i] = xs[i] @ W[e]   for every row i in chunk e.
-- This is the kernel underneath dmoe once tokens are already grouped by
-- expert; routing, grouping, activation and scaling are all stripped out.

#[blackbox(matmul)]
def matmul [n] [m] [p] (a: [n][m]f32) (b: [m][p]f32) : [n][p]f32 =
  ???

#[blackbox(irreg_matmul)]
def irreg_matmul [n] [m] [p] (a: [n][m]f32) (b: [m][p]f32) : [n][p]f32 =
  ???

-- Apply one expert to its slice of the grouped tokens (size g).
def apply_group [m] [E] [d1] [d2]
                (xs: [m][d1]f32) (offsets: [E]i64) (group_sizes: [E]i64)
                (W: [E][d1][d2]f32) (e: i64) : ?[g].([g][d2]f32, ()) =
  let g = group_sizes[e]
  let o = offsets[e]
  let chunk = take g (drop o xs)
  in (matmul chunk W[e], ())

def ragged_matmul [m] [E] [d1] [d2]
                  (offsets: [E]i64)
                  (group_sizes: [E]i64)
                  (xs: [m][d1]f32)
                  (W: [E][d1][d2]f32) : [m][d2]f32 =
  let (_shape, _flags, _offsets, y, _uniform) =
    flatmap (apply_group xs offsets group_sizes W) (iota E)
  in sized m y

-- Testing.
--
-- Two groups: row 0 uses the identity, rows 1-2 use 2*identity.
-- ==
-- entry: ragged_matmul_test
-- input {
--   [0i64, 1i64]
--   [1i64, 2i64]
--   [[1f32, 2f32], [3f32, 4f32], [5f32, 6f32]]
--   [[[1f32, 0f32], [0f32, 1f32]],
--    [[2f32, 0f32], [0f32, 2f32]]]
-- }
-- output {
--   [[1f32, 2f32], [6f32, 8f32], [10f32, 12f32]]
-- }
entry ragged_matmul_test [m] [E] [d1] [d2]
                         (offsets: [E]i64)
                         (group_sizes: [E]i64)
                         (xs: [m][d1]f32)
                         (W: [E][d1][d2]f32) : [m][d2]f32 =
  ragged_matmul offsets group_sizes xs W
