-- ==
-- input { 8i64 }
-- output { [8i64, 9i64, 10i64, 11i64, 12i64, 13i64, 14i64, 15i64] }
def main (n: i64) = #[unsafe] map (+n) (iota n)
