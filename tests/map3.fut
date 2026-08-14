-- ==
-- input { [[1i32, 2i32, 3i32], [4i32, 5i32, 6i32]] 100i32 }
-- output { [[101i32, 102i32, 103i32], [104i32, 105i32, 106i32]] }
def main (xss: [][]i32) (c: i32) = map (map (+c)) xss
