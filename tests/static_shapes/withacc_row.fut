-- with_acc-based row scatter: `scatter dest perm vals` writes whole rows
-- (`dest[perm[i]] = vals[i]`), exercising LowerWithAcc's window-dimension
-- case (updateWindowDims non-empty) rather than scatter.fut's plain
-- scalar case (LowerSegHist's own windowRank==0 special case of the same
-- formula).
-- ==
-- input {
--   [[0.0f32,0.0f32],[0.0f32,0.0f32],[0.0f32,0.0f32],[0.0f32,0.0f32]]
--   [3i64, 1i64, 0i64, 2i64]
--   [[1.0f32,2.0f32],[3.0f32,4.0f32],[5.0f32,6.0f32],[7.0f32,8.0f32]]
-- }
-- output { [[5.0f32,6.0f32],[3.0f32,4.0f32],[7.0f32,8.0f32],[1.0f32,2.0f32]] }
def main (dest: *[4][2]f32) (perm: [4]i64) (vals: [4][2]f32) : [4][2]f32 =
  scatter dest perm vals
