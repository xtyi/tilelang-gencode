import tilelang
import tilelang.language as T

tilelang.disable_cache()

M = 1024
N = 1024

def tma_add_one(M: int, N: int,
                BLOCK_M: int = 128, BLOCK_N: int = 128,
                dtype: str = "float32"):
    @T.prim_func
    def kernel(A: T.Tensor((M, N), dtype),
               C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=128) as (bx, by):
            tile = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)

            T.copy(A[by * BLOCK_M, bx * BLOCK_N], tile, disable_tma=False)

            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                tile[i, j] = tile[i, j] + 1

            T.copy(tile, C[by * BLOCK_M, bx * BLOCK_N], disable_tma=False)

    return kernel

program = tma_add_one(M, N)
kernel = tilelang.compile(program, out_idx=-1, target="musa", execution_backend="cython", verbose=True)
print(kernel.get_kernel_source())


import torch
A = torch.randn(M, N, device="musa", dtype=torch.float32)
C = kernel(A)
torch.testing.assert_close(C, A + 1, rtol=1e-6, atol=1e-6)
print("OK")
