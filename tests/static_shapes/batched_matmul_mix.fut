-- Batched matmul, custom flattening style -- STATIC variant of
-- dynamic_shapes/custom_flattening_matmul.fut's #[flatten_to_uniform(...)]
-- pattern. A mapped plain `matmul` is rewritten to a uniform batched matmul
-- via #[flatten_to_uniform($batched_matmul)]; the flatten target
-- (batched_matmul, a dummy) coerces the flatten-provided dynamic sizes to
-- literal sizes (b=n=k=m=2 here) and calls the real black box
-- batched_dot_general on statically-sized args, so the MLIR backend should
-- see a static dot_general -- same shape/data as dot_general_batched.fut,
-- just reached through the new custom-flattening machinery instead of a
-- bare dot_general call.

#[noinline] #[blackbox(dot_general)]
def batched_dot_general [b] [n] [k] [m] [nb] [nc]
                        (x: [b][n][k]f32)
                        (y: [b][k][m]f32)
                        (lhs_batching_dims: [nb]i64)
                        (rhs_batching_dims: [nb]i64)
                        (lhs_contracting_dims: [nc]i64)
                        (rhs_contracting_dims: [nc]i64) : *[b][n][m]f32 =
  #[unsafe] ???

#[blackbox(dummy)] #[noinline]
def batched_matmul [b] [n] [k] [m] (x: [b][n][k]f32) (y: [b][k][m]f32) : *[2][2][2]f32 =
  -- Coerce the flatten-provided (dynamic) sizes to the statically-known bench
  -- sizes b/n/k/m so the backend recovers static tensor shapes and lowers to a
  -- static dot_general. The result is already static, so it flows out without a
  -- (multi-dim) reshape -- the outer flatten glue coerces it back.
  let xc = #[unsafe] (x :> [2][2][2]f32)
  let yc = #[unsafe] (y :> [2][2][2]f32)
  in batched_dot_general xc yc [0i64] [0i64] [2i64] [1i64]

#[noinline] #[blackbox(matmul)]
def matmul [n] [k] [m] (x: [n][k]f32) (y: [k][m]f32) : *[n][m]f32 =
  #[unsafe] tabulate_2d n m (\i j -> ???)

def single [n] [k] [m] (x: [n][k]f32) (y: [k][m]f32) : [n][m]f32 =
  #[flatten_to_uniform($batched_matmul)] matmul x y

-- ==
-- input {
--   [[[1f32, 2f32], [3f32, 4f32]], [[5f32, 6f32], [7f32, 8f32]]]
--   [[[1f32, 0f32], [0f32, 1f32]], [[1f32, 0f32], [0f32, 1f32]]]
-- }
-- output {
--   [[[1f32, 2f32], [3f32, 4f32]], [[5f32, 6f32], [7f32, 8f32]]]
-- }
entry main (xs: [2][2][2]f32) (ys: [2][2][2]f32) : [2][2][2]f32 =
  #[unsafe]
  map2 single xs ys
