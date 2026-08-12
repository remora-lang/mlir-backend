#[noinline]
#[blackbox(matmul)]
def matmul_f32 [n][k][m] (x: [n][k]f32) (y: [k][m]f32): [n][m]f32 = ???

def main x y = matmul_f32 x y
