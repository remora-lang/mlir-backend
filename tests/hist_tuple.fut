-- Phase 7: tuple-operator histogram (running sum and running product into
-- two destination arrays from one seghist), gated off per the brief's
-- explicit "defer solving" instruction for SegHist's tuple case -- unlike
-- SegScan (Phase 6), no parallel-prefix-style workaround was in scope
-- here. Confirmed via the GPU IR dump that a tuple operator shows up as
-- HistOp.neutral/dest having 2 entries each (not a separate SegHist per
-- operator), which is what LowerSegHist's neutral.size() != 1 gate checks.
-- undefined
def main (is: [4]i64) (vs0: [4]i32) (vs1: [4]i32) : ([3]i32, [3]i32) =
  unzip (reduce_by_index (replicate 3 (0, 1))
                         (\(a0, a1) (b0, b1) -> (a0 + b0, a1 * b1))
                         (0, 1) is (zip vs0 vs1))
