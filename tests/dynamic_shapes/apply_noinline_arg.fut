-- Applying a function with a dynamically-sized array parameter to a
-- statically-shaped argument.
#[noinline]
def sum (xs: []i32): i32 = #[unsafe] reduce (+) 0 xs

-- ==
-- input { 5i32 }
-- output { 15i32 }
def main (n: i32) = sum [1,2,3,4] + n
