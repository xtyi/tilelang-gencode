import tilelang
import tilelang.language as T

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
}

@tilelang.jit(target="cuda --arch=sm_90", pass_configs=PASS_CONFIGS)
def tma_load_add_one(A, block_m, block_n, dtype="float32"):
    M, N = T.const("M N")
    A: T.Tensor[[M, N], dtype]
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=128) as (
        bx,
        by,
    ):
        tile = T.alloc_shared((block_m, block_n), dtype)
        T.copy(A[by * block_m, bx * block_n], tile)

        for i, j in T.Parallel(block_m, block_n):
            tile[i, j] = tile[i, j] + 1

        T.copy(tile, C[by * block_m, bx * block_n])
    return C


def main():

    M = 128
    N = 128
    block_m = 8
    block_n = 16
    kernel = tma_load_add_one.compile(M=M, N=N, block_m=block_m, block_n=block_n, dtype="float32")

    source = kernel.get_kernel_source()
    print(source)



if __name__ == "__main__":
    main()
