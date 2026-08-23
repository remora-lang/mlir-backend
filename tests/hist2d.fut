-- 2-D histogram via reduce_by_index_2d (SegHist with 2-D indices).
-- bins[is0[j]][is1[j]] += vs[j], over a 2x2 grid, starting from zeros.
-- The two index arrays are zipped into the (i64, i64) index the SOAC expects.

-- ==
-- input {
--   [0i64, 1i64, 0i64, 1i64]
--   [0i64, 1i64, 0i64, 0i64]
--   [5i32, 7i32, 3i32, 2i32]
-- }
-- output { [[8i32, 0i32], [2i32, 7i32]] }
def main [n] (is0: [n]i64) (is1: [n]i64) (vs: [n]i32) : [2][2]i32 =
  reduce_by_index_2d (replicate 2 (replicate 2 0i32)) (+) 0 (zip is0 is1) vs
