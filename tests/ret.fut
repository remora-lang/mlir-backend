-- Tests multiple returns with no segops (pure scalar body).
-- ==
-- input { 3i64 4i64 }
-- output { 7i64 12i64 }
def main (x: i64) (y: i64) = (x+y, x*y)
