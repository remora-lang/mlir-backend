-- Static-shape reduction for GPU benchmarking; the fixed size lets IREE apply
-- split-reduction (dynamic shapes can't).
-- ==
-- random input { [10000000]i32 }
def main (xs: [10000000]i32): i32 = #[unsafe] reduce (+) 0 xs
