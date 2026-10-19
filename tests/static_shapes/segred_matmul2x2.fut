-- 2x2 matrix product is associative but non-commutative. Phase 8 found
-- this exact case gave a numerically wrong result through
-- stablehlo.reduce before the fix: LowerSegRed's own reducer-body
-- block-arg binding was backwards (present since Phase 2, invisible
-- until a non-commutative operator exercised it, since every earlier
-- segred test was either commutative -- where the swap is
-- undetectable -- or gated to Undefined()). stablehlo.reduce itself
-- needed no workaround once the binding was corrected.
-- ==
-- input { [1i32, 2i32, 0i32] [0i32, 1i32, 1i32] [1i32, 0i32, 1i32] [1i32, 1i32, 2i32] }
-- output { 1i32 4i32 2i32 6i32 }
def mat2mul (a1,b1,c1,d1) (a2,b2,c2,d2) : (i32,i32,i32,i32) =
  (a1*a2 + b1*c2, a1*b2 + b1*d2, c1*a2 + d1*c2, c1*b2 + d1*d2)

def main (a: [3]i32) (b: [3]i32) (c: [3]i32) (d: [3]i32) =
  reduce mat2mul (1,0,0,1) (zip4 a b c d)
