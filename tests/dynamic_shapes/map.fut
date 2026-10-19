-- ==
-- input { [1i64, 2i64, 3i64, 4i64] }
-- output { [5i64, 6i64, 7i64, 8i64] }
def main [n] (xs: [n]i64) = map (+n) xs
