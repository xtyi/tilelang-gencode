import tilelang
import tilelang.language as T

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
}


@tilelang.jit(target="cuda --arch=sm_90", pass_configs=PASS_CONFIGS)
def tma_load_add_one_ws(A, block_m, block_k, dtype="float16"):
    M, K = T.const("M K")
    A: T.Tensor[[M, K], dtype]
    C = T.empty((M, K), dtype)

    with T.Kernel(T.ceildiv(M, block_m), threads=128) as by:
        tile = T.alloc_shared((block_m, block_k), dtype)

        for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=2):
            T.copy(
                A[by * block_m : (by + 1) * block_m, ko * block_k : (ko + 1) * block_k],
                tile,
            )

            for i, j in T.Parallel(block_m, block_k):
                tile[i, j] = tile[i, j] + 1

            T.copy(
                tile,
                C[by * block_m : (by + 1) * block_m, ko * block_k : (ko + 1) * block_k],
            )

    return C


def main():
    M = 8
    K = 256
    block_m = 4
    block_k = 128

    kernel = tma_load_add_one_ws.compile(
        M=M,
        K=K,
        block_m=block_m,
        block_k=block_k,
        dtype="float16",
    )

    source = kernel.get_kernel_source()
    print(source)


if __name__ == "__main__":
    main()
