-- Tuple-operator scan (running sum and running product together). This
-- particular operator is actually safe (both components are independent,
-- commutative combines), but segscan's current reduce_window-based
-- lowering conservatively defers *all* tuple operators to Phase 6 rather
-- than risk a case it can't distinguish from the genuinely unsafe ones --
-- see scan_noncomm.fut and LowerSegScan's comment for why.
-- undefined
def main (xs: [4]i32) =
  unzip (scan (\(a0,a1) (b0,b1) -> (a0+b0, a1*b1)) (0,1) (zip xs xs))
