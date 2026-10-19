-- Phase 7: out-of-bounds indices (none of the original hist1d/2d/3d.fut
-- exercise this). Index 5 (>= the 4-bin destination) and -1 (negative) are
-- both expected to be silently dropped, not written and not erroring --
-- confirmed to be Futhark's own `hist` semantics by running this exact
-- case through Futhark's reference interpreter first, and confirmed to be
-- stablehlo.scatter's own documented semantics by reading the spec's
-- `scatter` section directly ("If result_index is in bounds... Otherwise
-- updated_results = results"), not assumed to line up by coincidence.
-- ==
-- input { [0i64, 5i64, 1i64, -1i64] [10i32, 99i32, 2i32, 77i32] }
-- output { [10i32, 2i32, 0i32, 0i32] }
def main (is: [4]i64) (vs: [4]i32) : [4]i32 =
  hist (+) 0 4 is vs
