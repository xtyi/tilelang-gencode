import tilelang
import tilelang.language as T


def matmul(M, N, K, block_M, block_N, block_K, num_stages, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    arch = "sm_90"
    num_stages = 3
    WS = 1
    TMA = 1
    WGMMA = 1
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: not bool(WS),
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: not bool(TMA),
        tilelang.PassConfigKey.TL_DISABLE_WGMMA: not bool(WGMMA),
    }
    program = matmul(1024, 1024, 1024, 128, 128, 32, num_stages)
    kernel = tilelang.compile(program, target=f"cuda -arch={arch}", out_idx=[-1], pass_configs=pass_configs)
    with open(f"gemm_{arch}_ws{WS}_tma{TMA}_wgmma{WGMMA}_stage{num_stages}.cu", 'w') as f:
        f.write(kernel.get_kernel_source())
    # print(kernel.get_kernel_source())

main()
