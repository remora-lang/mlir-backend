-- A standalone top-level if (today's only other `if` usage is nested
-- inside a reduce lambda in argmax.fut/moe_black_box.fut). The condition is
-- computed from an integer comparison rather than a bare bool parameter,
-- since run_tests.py's literal parser doesn't support bool literals
-- (a separate, pre-existing gap, not a backend issue).
-- ==
-- input { 1i32 10i32 20i32 }
-- output { 11i32 }
-- input { 0i32 10i32 20i32 }
-- output { 19i32 }
def main (c: i32) (x: i32) (y: i32) : i32 =
  if c > 0 then x + 1 else y - 1
