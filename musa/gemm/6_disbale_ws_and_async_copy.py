import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TIR_USE_ASYNC_COPY: False,
}


@tilelang.jit(pass_configs=PASS_CONFIGS)
def gemm_ws(A, B, block_m, block_n, block_k, dtype="float16", accum_dtype="float32"):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_m, block_k), dtype)
        B_shared = T.alloc_shared((block_k, block_n), dtype)
        C_local = T.alloc_fragment((block_m, block_n), accum_dtype)

        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=2):
            T.copy(A[by * block_m, ko * block_k], A_shared)
            T.copy(B[ko * block_k, bx * block_n], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * block_m, bx * block_n])

    return C



def main():
    M = 1024
    N = 1024
    K = 1024
    block_m = 128
    block_n = 128
    block_k = 32

    kernel = gemm_ws.compile(
        M=M,
        N=N,
        K=K,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )

    source = kernel.get_kernel_source()
    print(source)

    A = torch.randn((M, K), dtype=torch.float16, device="musa")
    B = torch.randn((K, N), dtype=torch.float16, device="musa")
    C = kernel(A, B)
    torch.testing.assert_close(C, A @ B, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    main()
