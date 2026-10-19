-- Affine-map composition (associative, non-commutative) scanned instead of
-- reduced. Same non-commutative gate as segred_affine_compose.fut and for
-- the same reason -- despite scan being inherently order-dependent by
-- definition, XLA's actual stablehlo.reduce_window implementation does not
-- reliably preserve left-to-right combination order for a non-commutative
-- operator: an earlier version of this backend had no such gate for scan,
-- reasoning that reduce_window's structure made it safe, and this exact
-- test case silently produced a numerically wrong (not crashing) result on
-- XLA:CPU. See Phase 8.
-- undefined
def compose (a1,b1) (a2,b2) : (i32,i32) = (a1*a2, a1*b2+b1)

def main (a: [3]i32) (b: [3]i32) =
  unzip (scan compose (1,0) (zip a b))
