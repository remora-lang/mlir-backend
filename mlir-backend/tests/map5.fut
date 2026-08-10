-- Tests some iota IR construct.
def main (n: i64) =
  #[unsafe] map (\i -> (i, i+n)) (iota n)
