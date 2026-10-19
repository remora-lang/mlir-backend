-- ==
-- input { 4i64 }
-- output { [0i64, 1i64, 2i64, 3i64] }
def main (n: i64) : [n]i64 = #[unsafe] iota n
