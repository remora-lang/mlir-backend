-- Tests multiple returns.
-- ==
-- input { 4i64 }
-- output { [1i64, 2i64, 3i64, 4i64] [4i64, 5i64, 6i64, 7i64] }
def main (n: i64) =
  #[unsafe] unzip (map (\i -> (i+1, i+n)) (iota n))
