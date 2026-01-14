#include <tl_templates/musa/common.h>
#include <tl_templates/musa/intrin.h>
#include <tl_templates/musa/gemm.h>
#include <tl_templates/musa/copy.h>
#include <tl_templates/musa/barrier.h>
#include <tl_templates/musa/reduce.h>
#include <tl_templates/musa/threadblock_swizzle.h>
#include <tl_templates/musa/debug.h>

extern "C" __global__ void main_kernel();
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel() {
  __musa_async_bar_record(0);
  if (((int)threadIdx.x) == 0) {
    debug_print_var("print tid", ((int)threadIdx.x));
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';

    return 0;
}

extern "C" int call(musaStream_t stream=musaStreamDefault) {
        main_kernel<<<dim3(1, 1, 1), dim3(256, 1, 1), 0, stream>>>();
        TILELANG_CHECK_LAST_ERROR("main_kernel");

        return 0;
}
