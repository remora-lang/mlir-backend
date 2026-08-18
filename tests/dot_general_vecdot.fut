-- dot_general, 1-D case: [k] . [k] -> scalar.
-- Contracts lhs axis 0 with rhs axis 0; no batch, no free dimensions,
-- so the result has rank 0.

#[noinline]
#[blackbox(dot_general)]
def dot_general 'a 'b 'c [n] [m] (x: a) (y: b)
                (lhs_batching_dims: [n]i64) (rhs_batching_dims: [n]i64)
                (lhs_contracting_dims: [m]i64) (rhs_contracting_dims: [m]i64): c =
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
