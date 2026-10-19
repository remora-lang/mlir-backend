-- No StableHLO/XLA equivalent for an arbitrary-trip-count loop; this must
-- fail to compile with a named Undefined(), not crash or emit anything.
-- undefined
def main (n: i32) : i32 = loop x = 0 for i < n do x + i
