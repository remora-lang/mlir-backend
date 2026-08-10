def main (xss: [][]i32): []i32 = #[unsafe] map (reduce (+) 0) xss
