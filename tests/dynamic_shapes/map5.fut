-- Tests some iota IR construct.
-- ==
-- input { 4i64 }
-- output { [0i64, 1i64, 2i64, 3i64] [4i64, 5i64, 6i64, 7i64] }
def main (n: i64) =
  #[unsafe] unzip (map (\i -> (i, i+n)) (iota n))
