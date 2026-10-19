-- Affine-map composition (x -> a*x+b) is associative but non-commutative;
-- another clean segred non-commutative-operator stress test. See
-- segred_matmul2x2.fut. stablehlo.reduce's own spec says reduction order
-- is implementation-defined, which sounds like it would need commutativity
-- -- but the spec's actual requirement is just a (not-necessarily-
-- commutative) monoid, exactly what Futhark's own `neutral` already
-- guarantees. The real Phase 8 bug was LowerSegRed's own reducer-body
-- block-arg binding being backwards (present since Phase 2, invisible
-- until a non-commutative operator exercised it); fixed directly, no
-- workaround needed.
-- ==
-- input { [2i32, 3i32, 1i32] [1i32, 0i32, 5i32] }
-- output { 6i32 31i32 }
def compose (a1,b1) (a2,b2) : (i32,i32) = (a1*a2, a1*b2+b1)

def main (a: [3]i32) (b: [3]i32) =
  reduce compose (1,0) (zip a b)
