-- Segmented non-commutative reduce: combines reduce2.fut's segmentation
-- with segred_matmul2x2.fut's non-commutative operator, solved by Phase
-- 8's block-arg binding fix in LowerSegRed (plain stablehlo.reduce, no
-- scan-based workaround needed).
-- ==
-- input {
--   [[1i32, 2i32], [0i32, 1i32]]
--   [[0i32, 1i32], [1i32, 1i32]]
--   [[1i32, 0i32], [1i32, 0i32]]
--   [[1i32, 1i32], [2i32, 1i32]]
-- }
-- output { [2i32, 0i32] [1i32, 1i32] [2i32, 1i32] [2i32, 3i32] }
def mat2mul (a1,b1,c1,d1) (a2,b2,c2,d2) : (i32,i32,i32,i32) =
  (a1*a2 + b1*c2, a1*b2 + b1*d2, c1*a2 + d1*c2, c1*b2 + d1*d2)

def main (a: [2][2]i32) (b: [2][2]i32) (c: [2][2]i32) (d: [2][2]i32) =
  #[flattening(only_inner)]
  map4 (\ar br cr dr -> reduce mat2mul (1,0,0,1) (zip4 ar br cr dr)) a b c d
