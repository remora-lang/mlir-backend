-- T=1 (single-token) specialization of moe_black_box, for the MLIR backend
-- test suite. The MLIR backend currently aborts compiling this
-- (std::runtime_error: undefined at compiler.hpp:1070); Futhark's own backends
-- produce the reference output below. Identity activation, so the output is
-- exact and hand-checkable.
module bb = import "moe_black_box"

-- ==
-- entry: main
-- input {
--   [[1.0f32, 0.0f32]]
--   [[1.0f32, 0.0f32], [0.0f32, 1.0f32]]
--   [[[1.0f32, 1.0f32], [1.0f32, 1.0f32]],
--    [[1.0f32, 1.0f32], [1.0f32, 1.0f32]]]
--   [[[1.0f32, 1.0f32], [1.0f32, 1.0f32]],
--    [[1.0f32, 1.0f32], [1.0f32, 1.0f32]]]
-- }
-- output {
--   [[2.0f32, 2.0f32]]
-- }
entry main (tokens: [1][2]f32) (W_router: [2][2]f32)
           (W_expert_in: [2][2][2]f32) (W_expert_out: [2][2][2]f32) : [1][2]f32 =
  #[flattening(only_inner)]
  bb.dmoe_top1 tokens W_router W_expert_in W_expert_out (\(x: f32) -> x)
