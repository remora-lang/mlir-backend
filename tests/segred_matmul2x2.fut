-- 2x2 matrix product is associative but non-commutative. The StableHLO
-- spec says reduce preserves element order, but XLA's GPU reduce may
-- assume commutativity (see Phase 8) -- so this is Undefined() for now
-- rather than risking a silently wrong result.
-- undefined
def mat2mul (a1,b1,c1,d1) (a2,b2,c2,d2) : (i32,i32,i32,i32) =
  (a1*a2 + b1*c2, a1*b2 + b1*d2, c1*a2 + d1*c2, c1*b2 + d1*d2)

def main (a: [3]i32) (b: [3]i32) (c: [3]i32) (d: [3]i32) =
  reduce mat2mul (1,0,0,1) (zip4 a b c d)
