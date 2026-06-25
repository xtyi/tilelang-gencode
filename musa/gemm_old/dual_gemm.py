import torch
import tilelang
import tilelang.language as T

tilelang.disable_cache()


TARGET = "musa"
DEVICE = "musa"


def dual_gemm(
    seq_q,
    seq_kv,
    dim,
    block_M=256,
    block_N=64,
    num_stages=2,
    threads=512,
):
    q_shape = [seq_q, dim]
    kv_shape = [seq_kv, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(
                T.ceildiv(seq_q, block_M), threads=threads) as bx:
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            P_shared = T.alloc_shared([block_M, block_N], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)

            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)

            T.copy(Q[bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)

            for k in T.Pipelined(T.ceildiv(seq_kv, block_N), num_stages=num_stages):

                T.copy(K[k * block_N:(k + 1) * block_N, :], K_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

                T.copy(acc_s, P_shared)

                T.copy(V[k * block_N:(k + 1) * block_N, :], V_shared)
                T.gemm(P_shared, V_shared, acc_o)


            T.copy(acc_o, Output[bx * block_M:(bx + 1) * block_M, :])

    return main


def ref_program(Q, K, V):
    scores = torch.einsum("qd,kd->qk", Q, K)
    output = torch.einsum("qk,kd->qd", scores, V)

    return output


seq_q = 256
seq_kv = 128
dim = 128

program = dual_gemm(
    seq_q,
    seq_kv,
    dim,
)

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
    tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True,
}

kernel = tilelang.compile(
    program,
    out_idx=[-1],
    target=TARGET,
    execution_backend="cython",
    verbose=True,
    pass_configs=pass_configs,
)

print(kernel.get_kernel_source())

dtype = "float16"
q = torch.rand(seq_q, dim, device=DEVICE, dtype=getattr(torch, dtype))
k = torch.rand(seq_kv, dim, device=DEVICE, dtype=getattr(torch, dtype))
v = torch.rand(seq_kv, dim, device=DEVICE, dtype=getattr(torch, dtype))

output = kernel(q, k, v)
ref_output = ref_program(q, k, v)

torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
print("All checks pass.")
