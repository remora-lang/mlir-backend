-- Ragged_dot, combinator style, custom flattening -- STATIC variant.
-- Same mode-1 ragged matmul as ragged_dot_mode1.fut
-- ([m][k] . [g][k][n] . [g] -> [m][n], group_sizes partitions the m rows
-- of lhs into g contiguous groups), but reached through the new
-- flatmap/#[flatten_to_nonuniform(...)] custom-flattening machinery
-- instead of a bare ragged_dot call. g=2, k=2, n=2, group_sizes=[1,3]
-- (so m=4, a_k=m*k=8, b_k=g*k*n=8) are filled in below as literals (the
-- backend recovers static sizes from literal shapes, not from the
-- hoisted top-level `let g` etc).
--
-- The flatten target (matmul_nonuniform, a dummy) coerces the
-- flatten-provided dynamic sizes to these statically-known sizes and
-- calls the real black box matmul_nonuniform_bb on statically-sized
-- args. The MLIR backend then sees a static a_k -> static m and should
-- lower to chlo.ragged_dot.
--
-- Same lhs/rhs/group_sizes as ragged_dot_mode1.fut's first case (just f32
-- instead of i64), so the expected output is already known.

-- Inputs
-- ------
-- 0: g          number of segments
-- 1: ms         [g] group sizes (== ragged_dot's group_sizes)
-- 2: ks         [g] replicate(k) -- k uniform across groups
-- 3: ns         [g] replicate(n) -- n uniform across groups
-- 4: a_k        flat lhs length (dynamic)
-- 5..7:         a_segment_sizes / a_flag / a_offsets (unused)
-- 8: a_data     flat lhs [a_k]         (dynamic in m)
-- 9: b_k        flat rhs length (static, = g*k*n)
-- 10..12:       b_segment_sizes / b_flag / b_offsets (unused)
-- 13: b_data    flat rhs [b_k]         (static)
--
-- Outputs: the existential size r followed by the segmented tuple
--   (r : i64, sizes : [g]i64, flags : [r]i1, offsets : [g]i64, data :
--   [r]f32)
-- Only `data` (the flattened [m][n] result) is meaningful; the auxiliary
-- segment structures are dummies for this demonstration.
#[blackbox(ragged_dot_nonuniform)] #[noinline]
def matmul_nonuniform_bb (g: i64)
                         (ms: [g]i64)
                         (ks: [g]i64)
                         (ns: [g]i64)
                         (a_k: i64)
                         (a_segment_sizes: [g]i64)
                         (a_flag: [a_k]bool)
                         (a_offsets: [g]i64)
                         (a_data: [a_k]f32)
                         (b_k: i64)
                         (b_segment_sizes: [g]i64)
                         (b_flag: [b_k]bool)
                         (b_offsets: [g]i64)
                         (b_data: [b_k]f32) : ?[r].(*[g]i64, *[r]bool, *[g]i64, *[r]f32) =
  let r = i64.sum (map2 (*) ms ns)
  in (replicate g 0, replicate r true, replicate g 0, replicate r 0.0)

#[noinline] #[blackbox(matmul)]
def matmul (m: i64) (k: i64) (n: i64) (a: [m][k]f32) (b: [k][n]f32) : [m][n]f32 =
  map (\row -> map (\col -> f32.sum (map2 (*) row col)) (transpose b)) a

-- Apply group i to its slice of the rows.
def apply_group [m] [g] [k] [n]
                (lhs: [m][k]f32)
                (rhs: [g][k][n]f32)
                (group_sizes: [g]i64)
                (offsets: [g]i64)
                (i: i64) : ?[s].([s][n]f32, ()) =
  let s = group_sizes[i]
  let o = offsets[i]
  let chunk = take s (drop o lhs)
  in (#[flatten_to_nonuniform($matmul_nonuniform_bb)] matmul s k n chunk rhs[i], ())

def ragged_matmul [m] [g] [k] [n]
                  (lhs: [m][k]f32)
                  (rhs: [g][k][n]f32)
                  (group_sizes: [g]i64)
                  (offsets: [g]i64) : [m][n]f32 =
  let (_shape, _flags, _offsets, y, _uniform) =
    #[flattening(only_inner)]
    flatmap (apply_group lhs rhs group_sizes offsets) (iota g)
  in sized m y

-- ==
-- input {
--   [[1f32, 0f32], [1f32, 1f32], [2f32, 1f32], [0f32, 3f32]]
--   [[[1f32, 0f32], [0f32, 1f32]], [[1f32, 1f32], [0f32, 1f32]]]
--   [1i64, 3i64]
--   [0i64, 1i64]
-- }
-- output {
--   [[1f32, 0f32], [1f32, 2f32], [2f32, 3f32], [0f32, 3f32]]
-- }
entry main (lhs: [4][2]f32)
           (rhs: [2][2][2]f32)
           (group_sizes: [2]i64)
           (offsets: [2]i64) : [4][2]f32 =
  #[flattening(only_inner)] #[unsafe]
  ragged_matmul lhs rhs group_sizes offsets
