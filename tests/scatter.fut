-- ==
-- input { [3i64, 2i64, 1i64, 0i64, -1i64] [1i64, 2i64, 3i64, 4i64, 100i64] }
-- output { [4i64, 3i64, 2i64, 1i64] }
def main (is: [5]i64) (xs: [5]i64): []i64 =
  scatter (replicate 4 0) is xs
