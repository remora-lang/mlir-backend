-- Custom flattening: a mapped `matmul` is rewritten to a uniform batched
-- matmul via #[flatten_to_uniform($batched_matmul)], where batched_matmul is a
-- wrapper around a batched dot_general. Exercises the dynamic `replicate` that
-- flattening inserts for the shared W_router.
--
-- W_router is the identity, so routing_probabilities == tokens for every batch.

#[noinline] #[blackbox(dot_general)]
def batched_dot_general [b] [n] [k] [m] [nb] [nc]
                        (x: [b][n][k]f32)
                        (y: [b][k][m]f32)
                        (lhs_batching_dims: [nb]i64)
                        (rhs_batching_dims: [nb]i64)
                        (lhs_contracting_dims: [nc]i64)
                        (rhs_contracting_dims: [nc]i64) : *[b][n][m]f32 =
  #[unsafe] ???

#[blackbox(dummy)]
def batched_matmul [b] [n] [k] [m] (x: [b][n][k]f32) (y: [b][k][m]f32) : *[b][n][m]f32 =
  batched_dot_general x y [0i64] [0i64] [2i64] [1i64]

#[noinline] #[blackbox(matmul)]
def matmul [n] [k] [m] (x: [n][k]f32) (y: [k][m]f32) : *[n][m]f32 =
  #[unsafe] tabulate_2d n m (\i j -> ???)

def dmoe [T] [E] [d1] [d2]
         (tokens: [T][d1]f32)
         (W_router: [d1][E]f32)
         (W_expert_in: [E][d1][d2]f32)
         (W_expert_out: [E][d2][d1]f32) =
  let routing_probabilities: [T][E]f32 = #[flatten_to_uniform($batched_matmul)] matmul tokens W_router
  in routing_probabilities

-- ==
-- input {
--   [[[1f32, 2f32]], [[3f32, 4f32]]]
--   [[1f32, 0f32], [0f32, 1f32]]
--   [[[1f32, 1f32], [1f32, 1f32]], [[1f32, 1f32], [1f32, 1f32]]]
--   [[[1f32, 1f32], [1f32, 1f32]], [[1f32, 1f32], [1f32, 1f32]]]
-- }
-- output {
--   [[[1f32, 2f32]], [[3f32, 4f32]]]
-- }
entry main [B] [T] [E] [d1] [d2]
           (tokens: [B][T][d1]f32)
           (W_router: [d1][E]f32)
           (W_expert_in: [E][d1][d2]f32)
           (W_expert_out: [E][d2][d1]f32) =
  map (\ts -> dmoe ts W_router W_expert_in W_expert_out) tokens
