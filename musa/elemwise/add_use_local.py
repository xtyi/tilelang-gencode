import tilelang
import tilelang.language as T
import torch


def elementwise_add(N, num_per_thread=8, threads=256, dtype="float32"):
    num_per_block = threads * num_per_thread

    @T.prim_func
    def main(A: T.Tensor((N), dtype), B: T.Tensor((N), dtype), C: T.Tensor((N), dtype)):
        with T.Kernel(T.ceildiv(N, num_per_block), threads=threads) as (bx):
            # vector add.
            for i in T.Parallel(threads):
                # 使用 local 则需要拆分 T.Parallel, 在线程级别里 alloc
                sum = T.alloc_local((num_per_thread), "float")
                for j in T.serial(num_per_thread):
                    offsets = i * num_per_thread + j
                    sum[j] = (
                        A[bx * num_per_block + offsets] + B[bx * num_per_block + offsets]
                    )
                T.copy(sum, C[bx * num_per_block + i * num_per_thread])

    return main



def ref_program(x, y):
    return x + y


N = 4096

program = elementwise_add(N)
kernel = tilelang.compile(
    program, out_idx=-1, target="musa", execution_backend="cython", verbose=True
)
print(kernel.get_kernel_source())

a = torch.randn(N, dtype=torch.float32, device="musa")
b = torch.randn(N, dtype=torch.float32, device="musa")

c = kernel(a, b)

torch.testing.assert_close(c, ref_program(a, b), rtol=1e-2, atol=1e-2)
