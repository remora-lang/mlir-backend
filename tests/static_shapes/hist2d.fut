-- Phase 7: static-shape version of hist2d.fut, exercising LowerSegHist's
-- multi-dimensional index path (stacking 2 separate index-component arrays
-- into one [n, 2] stablehlo.scatter indices tensor via concatenate), which
-- hist_static.fut's 1-D case can't exercise (there, indexRank == 1 skips
-- the concatenate entirely). See hist_static.fut for why the original
-- hist2d.fut (dynamic shape) can't exercise this backend at all.
-- ==
-- input {
--   [0i64, 1i64, 0i64, 1i64]
--   [0i64, 1i64, 0i64, 0i64]
--   [5i32, 7i32, 3i32, 2i32]
-- }
-- output { [[8i32, 0i32], [2i32, 7i32]] }
def main (is0: [4]i64) (is1: [4]i64) (vs: [4]i32) : [2][2]i32 =
  reduce_by_index_2d (replicate 2 (replicate 2 0i32)) (+) 0 (zip is0 is1) vs
