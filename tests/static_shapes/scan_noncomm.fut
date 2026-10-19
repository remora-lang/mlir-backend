-- Affine-map composition (associative, non-commutative) scanned instead of
-- reduced. Lowered via the Phase 6 parallel-prefix algorithm
-- (jax.lax.associative_scan's slice/pad/concatenate recursion, ported
-- directly from its Python source) -- verified to preserve combination
-- order correctly, unlike stablehlo.reduce_window, which silently
-- computed the wrong answer for this same operator (see
-- segred_affine_compose.fut for the reduce, not scan, version -- that one
-- still hits Undefined() for the unrelated, real reason that segred's
-- commutativity gate is accurate).
-- ==
-- input { [2i32, 3i32, 1i32] [1i32, 0i32, 5i32] }
-- output { [2i32, 6i32, 6i32] [1i32, 1i32, 31i32] }
def compose (a1,b1) (a2,b2) : (i32,i32) = (a1*a2, a1*b2+b1)

def main (a: [3]i32) (b: [3]i32) =
  unzip (scan compose (1,0) (zip a b))
