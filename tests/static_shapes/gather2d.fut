-- Row-gather: `tokens[perm[i], :]` -- one gathered dimension (0, via
-- `perm`) plus one kept dimension (1, the whole row), exactly the shape
-- `moe_black_box.fut`'s `grouped_tokens = map (\i -> tokens[i,:]) perm`
-- needs. See gather1d.fut's comment for why this needs a real
-- stablehlo.gather rather than the plain affine-read path.
-- ==
-- input {
--   [[1.0f32,2.0f32],[3.0f32,4.0f32],[5.0f32,6.0f32],[7.0f32,8.0f32]]
--   [3i64, 1i64, 0i64, 2i64]
-- }
-- output { [[7.0f32,8.0f32],[3.0f32,4.0f32],[1.0f32,2.0f32],[5.0f32,6.0f32]] }
def main (tokens: [4][2]f32) (perm: [4]i64) : [4][2]f32 =
  map (\i -> #[unsafe] tokens[i, :]) perm
