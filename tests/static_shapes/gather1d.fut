-- A gather: `arr[perm[i]]` inside a map, where the index comes from
-- another array rather than being a bare iteration variable --
-- `LowerSegMap`'s plain affine-read path can't express this (no static
-- affine map from iteration ids to the index), so it goes through a real
-- stablehlo.gather built outside the map entirely (a map region only ever
-- exposes each operand's per-element scalar to its block args, never a
-- whole array, so there's no way to do this gather's dynamic_slice
-- *inside* the map region). See gather2d.fut for the row-gather variant
-- `moe_black_box.fut` actually needs.
-- ==
-- input { [10i32, 20i32, 30i32, 40i32] [3i64, 1i64, 0i64, 2i64] }
-- output { [40i32, 20i32, 10i32, 30i32] }
def main (arr: [4]i32) (perm: [4]i64) : [4]i32 =
  map (\i -> #[unsafe] arr[i]) perm
