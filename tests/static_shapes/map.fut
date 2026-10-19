-- ==
-- input { 10i32 }
-- output { [11i32, 12i32, 13i32, 14i32] }
def main (n: i32) = map (+n) [1,2,3,4]
