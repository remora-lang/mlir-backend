-- ==
-- input { [1i32, 2i32, 3i32, 4i32] }
-- output { [1i32, 3i32, 6i32, 10i32] }
def main (xs: []i32): []i32 = scan (+) 0 xs
