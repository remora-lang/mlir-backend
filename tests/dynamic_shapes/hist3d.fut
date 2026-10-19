-- 3-D histogram via reduce_by_index_3d (SegHist with 3-D indices).
-- bins[is0[j]][is1[j]][is2[j]] += vs[j], over a 2x2x2 grid, from zeros.

-- ==
-- input {
--   [0i64, 1i64, 0i64, 0i64]
--   [0i64, 1i64, 0i64, 1i64]
--   [0i64, 1i64, 0i64, 0i64]
--   [1i32, 2i32, 3i32, 4i32]
-- }
-- output { [[[4i32, 0i32], [4i32, 0i32]], [[0i32, 0i32], [0i32, 2i32]]] }
def main [n] (is0: [n]i64) (is1: [n]i64) (is2: [n]i64) (vs: [n]i32) : [2][2][2]i32 =
  reduce_by_index_3d
    (replicate 2 (replicate 2 (replicate 2 0i32))) (+) 0
    (zip3 is0 is1 is2) vs
