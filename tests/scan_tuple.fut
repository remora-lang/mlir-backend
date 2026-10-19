-- Tuple-operator scan (running sum and running product together), no
-- existing coverage before this. Lowered via the Phase 6 parallel-prefix
-- algorithm (jax.lax.associative_scan's slice/pad/concatenate recursion),
-- not stablehlo.reduce_window -- the latter can't safely handle tuple
-- operators (see scan_noncomm.fut and LowerSegScan's comment).
-- ==
-- input { [1i32, 2i32, 3i32, 4i32] }
-- output { [1i32, 3i32, 6i32, 10i32] [1i32, 2i32, 6i32, 24i32] }
def main (xs: [4]i32) =
  unzip (scan (\(a0,a1) (b0,b1) -> (a0+b0, a1*b1)) (0,1) (zip xs xs))
