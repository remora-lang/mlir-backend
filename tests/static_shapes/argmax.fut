-- Phase 8: static-shape version of dynamic_shapes/argmax.fut (NOTE: the
-- operator is not commutative on ties). dynamic_shapes/argmax.fut itself
-- declares `xs: []f32` (dynamic size), so it hits the "dynamic-shaped
-- function parameter" Undefined() gate before LowerSegRed's own lowering
-- ever runs -- a permanent, unrelated limitation of this backend (same
-- bucket as dynamic_shapes/map.fut/hist1d.fut), not something Phase 8
-- fixes. Re-enabling dynamic_shapes/argmax.fut itself is therefore not
-- reachable; this statically-shaped equivalent is what actually exercises
-- Phase 8's fix.
--
-- argmax.fut's own two cases have no tie at the maximum, so they'd pass
-- even with a (wrong) commutativity assumption; the second case here adds
-- an actual tie (two 9s) to stress-test order preservation specifically
-- -- confirmed against Futhark's own reference interpreter that the
-- *later* occurrence wins a tie (the lambda's `else` branch takes the new
-- value on x == y), the opposite of what a naive "first occurrence wins"
-- guess would produce.
-- ==
-- input { [3f32, 1f32, 4f32, 1f32, 5f32, 9f32, 2f32] }
-- output { 9f32 5i64 }
-- input { [1f32, 9f32, 2f32, 9f32, 3f32, 0f32, 0f32] }
-- output { 9f32 3i64 }
def f32_argmax (xs: [7]f32) =
  reduce (\(x, i) (y, j) -> if x > y then (x, i) else (y, j))
         (f32.lowest, -1)
         (zip xs (indices xs))

def main (xs: [7]f32): (f32, i64) = f32_argmax xs
