-- argsort: the permutation of indices that sorts x ascending.

#[noinline] #[blackbox(argsort)]
def argsort [n] (x: [n]i32) (is_stable: bool) : [n]i64 =
  ???

-- ==
-- input { [5i32, 3i32, 8i32, 1i32, 9i32, 2i32, 7i32] }
-- output { [3i64, 5i64, 1i64, 0i64, 6i64, 2i64, 4i64] }
def main (x: [7]i32) : [7]i64 = argsort x false
