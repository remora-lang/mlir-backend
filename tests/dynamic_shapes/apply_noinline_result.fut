-- Applying a noinline function whose result is a dynamically-sized array, at a
-- call site where the result size is statically known.
#[noinline]
def addone (xs: []i32): []i32 = #[unsafe] map (+1) xs

-- ==
-- input { 10i32 }
-- output { [12i32, 13i32, 14i32, 15i32] }
def main (n: i32) = map (+n) (addone [1,2,3,4])
