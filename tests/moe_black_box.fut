-- Implementation of dropless mixture-of-experts.
--
-- Reference: MegaBlocks, Trevor Gale, arXiv:2211.15841
--
-- See Figure 1 in particular.
--

#[noinline] #[blackbox(scatter)]
def scatter 't [k] [n] (dest: [k]t) (is: [n]i64) (vs: [n]t) : [k]t =
  #[unsafe] ???

#[noinline] #[blackbox(matmul)]
def matmul 't [n] [k] [m] (x: [n][k]t) (y: [k][m]t) : [n][m]t =
  #[unsafe] ???

#[noinline] #[blackbox(ragged_dot)]
def ragged_dot 'a 'b 'c [g] [n] [m] [s]
               (lhs: a)
               (rhs: b)
               (group_sizes: [g]i64)
               (lhs_batching_dims: [n]i64)
               (rhs_batching_dims: [n]i64)
               (lhs_contracting_dims: [m]i64)
               (rhs_contracting_dims: [m]i64)
               (lhs_ragged_dim: i64)
               (rhs_group_dims: [s]i64) : c =
  #[unsafe] ???

#[noinline] #[blackbox(argsort)]
def argsort 'a [n] (x: [n]a) (is_stable: bool) : [n]i64 =
  ???

def ragged_matmul [m] [E] [d1] [d2]
                  (xs: [m][d1]f32)
                  (W: [E][d1][d2]f32)
                  (group_sizes: [E]i64) : [m][d2]f32 =
  ragged_dot xs W group_sizes [] [] [1i64] [1i64] 0i64 [0i64]

def f32_argmax xs =
  reduce (\(x, i) (y, j) -> if x > y then (x, i) else (y, j))
         (f32.lowest, -1)
         (zip xs (indices xs))

-- This is a simplified version where each token is routed
-- to only the most likely expert (top 1) rather than the
-- k most likely (top k).
--
-- Example parameters from the reference:
--   E = 64, k = 1, T = 1024,
--   d1 = [512, 768, 1024, 1536, 2048] (pick one; Table 1),
--   d2 = 4 * d1
--   * (batch size 512; dmoe is mapped over it)
-- Additional params:
--   * activation: gelu_approximate_f32 (paper doesn't specify)
--
-- NOTES on the implementation
-- - scaling could be done in the loop body with an sgemm primitive?
--   (Just remember to permute gate first.)
-- - argsort could be replaced by an E-way partition.
--
--
def dmoe_top1 [T] [E] [d1] [d2]
              (tokens: [T][d1]f32)
              (W_router: [d1][E]f32)
              (W_expert_in: [E][d1][d2]f32)
              (W_expert_out: [E][d2][d1]f32)
              (activation: f32 -> f32) =
  -- Route tokens to experts.
  let routing_probabilities: [T][E]f32 = matmul tokens W_router
  let (gate: [T]f32, expert_indices: [T]i64) =
    unzip (map f32_argmax routing_probabilities)
  -- Group tokens by expert.
  let perm: [T]i64 = argsort expert_indices false
  let grouped_tokens: [T][d1]f32 = map (\i -> #[unsafe] tokens[i, :]) perm
  let group_sizes: [E]i64 = hist (+) 0 E expert_indices (replicate T 1)
  -- Apply experts.
  let h1 = ragged_matmul grouped_tokens W_expert_in group_sizes
  let h1 = map (map activation) h1
  let h2 = ragged_matmul h1 W_expert_out group_sizes
  -- Undo grouping and scale.
  let y = scatter (unflatten (#[scratch] replicate (T * d1) 0f32)) perm h2
  in map2 (\g -> map (* g)) gate y

-- entry main (tokens: [1024][512]f32)
--            (W_router: [512][64]f32)
--            (W_expert_in: [64][512][2048]f32)
--            (W_expert_out: [64][2048][512]f32) : [1024][512]f32 =
--   #[flattening(only_inner)]
--   dmoe_top1 tokens W_router W_expert_in W_expert_out (\(x: f32) -> x)

-- Testing.
--
-- ==
-- input {
--   [[1.0f32, 0.0f32], [0.0f32, 2.0f32]]
--   [[1.0f32, 0.0f32], [0.0f32, 1.0f32]]
--   [[[1.0f32, 1.0f32], [1.0f32, 1.0f32]],
--    [[1.0f32, 1.0f32], [1.0f32, 1.0f32]]]
--   [[[1.0f32, 1.0f32], [1.0f32, 1.0f32]],
--    [[1.0f32, 1.0f32], [1.0f32, 1.0f32]]]
-- }
-- output {
--   [[2.0f32, 2.0f32], [8.0f32, 8.0f32]]
-- }
entry main (tokens: [2][2]f32)
           (W_router: [2][2]f32)
           (W_expert_in: [2][2][2]f32)
           (W_expert_out: [2][2][2]f32) : [2][2]f32 =
  #[flattening(only_inner)]
  dmoe_top1 tokens W_router W_expert_in W_expert_out (\(x: f32) -> x)
