-- NOTE: the operator is not commutative on ties.
def f32_argmax xs =
  reduce (\(x, i) (y, j) -> if x > y then (x, i) else (y, j))
         (f32.lowest, -1)
         (zip xs (indices xs))

-- ==
-- input { [3f32, 1f32, 4f32, 1f32, 5f32, 9f32, 2f32] }
-- output { 9f32 5i64 }
-- ==
-- input { [7f32, 2f32, 1f32, 4f32] }
-- output { 7f32 0i64 }
def main (xs: []f32): (f32, i64) = f32_argmax xs
