import tilelang
import tilelang.language as T

def debug_demo(threads=256):

    @T.prim_func
    def main():
        with T.Kernel(1, threads=threads):
            tid = T.get_thread_binding()
            if tid == 0:
                T.print(tid, "print tid")

    return main


program = debug_demo()
kernel = tilelang.compile(program, target="musa", execution_backend="cython", verbose=True)
print(kernel.get_kernel_source())
kernel()
