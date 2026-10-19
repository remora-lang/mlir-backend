-- Affine-map composition (x -> a*x+b) is associative but non-commutative;
-- another clean segred non-commutative-operator stress test. See
-- segred_matmul2x2.fut and Phase 8.
-- undefined
def compose (a1,b1) (a2,b2) : (i32,i32) = (a1*a2, a1*b2+b1)

def main (a: [3]i32) (b: [3]i32) =
  reduce compose (1,0) (zip a b)
