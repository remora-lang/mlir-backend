def sqrt_2_over_pi = f32.sqrt (2 / f32.pi)

def gelu_approximate_f32 x =
  -- Copied from:
  -- https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L438-L471
  let cdf = 0.5 * (1.0 + f32.tanh (sqrt_2_over_pi * (x + 0.044715 * (x ** 3))))
  in x * cdf

-- ==
-- input { [0f32, 1f32, -1f32] }
-- output { [0f32, 0.841192f32, -0.158808f32] }
def main (xs: []f32): []f32 = map gelu_approximate_f32 xs
