-- Tests multiple returns.
def main (n: i64) =
  #[unsafe] map (\i -> (i+1, i+n)) (iota n)
