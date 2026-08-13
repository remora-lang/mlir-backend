-- ==
-- input { [[1i32, 2i32, 3i32], [4i32, 5i32, 6i32]] }
-- output { [6i32, 15i32] }
 
def main (xss: [][]i32): []i32 =
  #[flattening(only_inner)]
  map (reduce (+) 0) xss
