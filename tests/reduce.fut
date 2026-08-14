-- ==
-- input { [1i32, 2i32, 3i32, 4i32] }
-- output { 10i32 }
def main (xs: []i32): i32 = #[unsafe] reduce (+) 0 xs
