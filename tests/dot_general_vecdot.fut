-- dot_general, 1-D case: [k] . [k] -> scalar.
-- Contracts lhs axis 0 with rhs axis 0; no batch, no free dimensions,
-- so the result has rank 0.

#[noinline]
#[blackbox(dot_general)]
def dot_general [k] [nb] [nc] (x: [k]i64) (y: [k]i64)
                (lhs_batching_dims: [nb]i64) (rhs_batching_dims: [nb]i64)
                (lhs_contracting_dims: [nc]i64) (rhs_contracting_dims: [nc]i64): i64 =
  #[unsafe] ???

-- ==
-- input {
--   [1i64, 2i64, 3i64]
--   [4i64, 5i64, 6i64]
-- }
-- output {
--   32i64
-- }
-- No batch dims; contract lhs axis 0 with rhs axis 0.
def main (x: [3]i64) (y: [3]i64) : i64 =
  dot_general x y [] [] [0i64] [0i64]
