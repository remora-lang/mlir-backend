-- Tests multiple returns through a reduce (fused tuple reduction).
-- Returns (sum xs, sum of doubled xs).
def main (xs: []i64) =
  #[unsafe] reduce (\(a0,a1) (b0,b1) -> (a0+b0, a1+b1)) (0i64,0i64) (map (\x -> (x, x+x)) xs)
