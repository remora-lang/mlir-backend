-- Ragged_dot, combinator style, nonuniform, lowered via a ragged_dot_nonuniform
-- black box. Mode 1: the ragged dimension is a lhs non-contracting dim.
--   [m][k] . [g][k][n] . [g] -> [m][n]
-- group_sizes partitions the m rows into g contiguous groups; rows in group i
-- are multiplied by rhs[i]. group_sizes is a runtime input, so m is dynamic and
-- this exercises the dense linalg.generic-mask + dot_general lowering path.

#[blackbox(ragged_dot_nonuniform)] #[noinline]
def matmul_nonuniform (g: i64)
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
  in (#[flatten_to_nonuniform($matmul_nonuniform)] matmul s k n chunk rhs[i], ())

def ragged_matmul [m] [g] [k] [n]
                  (lhs: [m][k]f32)
                  (rhs: [g][k][n]f32)
                  (group_sizes: [g]i64)
                  (offsets: [g]i64) : [m][n]f32 =
  let (_shape, _flags, _offsets, y, _uniform) =
    #[flattening(only_inner)]
    flatmap (apply_group lhs rhs group_sizes offsets) (iota g)
  in sized m y

-- m=4, k=2, n=3, g=2. group_sizes=[1,3]: row 0 -> rhs[0], rows 1..3 -> rhs[1].
-- ==
-- input {
--   [[1f32, 2f32], [3f32, 4f32], [5f32, 6f32], [7f32, 8f32]]
--   [[[1f32, 0f32, 0f32], [0f32, 1f32, 0f32]], [[1f32, 1f32, 1f32], [1f32, 1f32, 1f32]]]
--   [1i64, 3i64]
--   [0i64, 1i64]
-- }
-- output {
--   [[1f32, 2f32, 0f32], [7f32, 7f32, 7f32], [11f32, 11f32, 11f32], [15f32, 15f32, 15f32]]
-- }
entry main (lhs: [4][2]f32)
           (rhs: [2][2][3]f32)
           (group_sizes: [2]i64)
           (offsets: [2]i64) : [4][3]f32 =
  #[flattening(only_inner)] #[unsafe]
  ragged_matmul lhs rhs group_sizes offsets
