-- Phase 7: static-shape version of hist1d.fut's scalar scatter-add
-- histogram, lowered via stablehlo.scatter. hist1d.fut itself can never
-- exercise LowerSegHist at all -- it declares [n]i64/[n]i32 parameters
-- (dynamic size), so it hits the "dynamic-shaped function parameter"
-- Undefined() gate (same bucket as map2.fut/map_dynamic.fut/
-- iota_dynamic.fut -- a pre-existing, permanent limitation of this
-- StableHLO/XLA-targeted backend, not something Phase 7 fixes) before
-- SegHist's own lowering ever runs.
-- ==
-- input {
--   [0i64, 1i64, 1i64, 3i64, 1i64]
--   [10i32, 1i32, 2i32, 5i32, 3i32]
-- }
-- output { [10i32, 6i32, 0i32, 5i32] }
def main (is: [5]i64) (vs: [5]i32) : [4]i32 =
  hist (+) 0 4 is vs
