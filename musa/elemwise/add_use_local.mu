#include <tl_templates/musa/common.h>
#include <tl_templates/musa/intrin.h>
#include <tl_templates/musa/gemm.h>
#include <tl_templates/musa/copy.h>
#include <tl_templates/musa/barrier.h>
#include <tl_templates/musa/reduce.h>
#include <tl_templates/musa/threadblock_swizzle.h>
#include <tl_templates/musa/debug.h>

extern "C" __global__ void main_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  __musa_async_bar_record(0);
  float sum[8];
  for (int i = 0; i < 256; ++i) {
    for (int j = 0; j < 2; ++j) {
      float4 __1;
        float4 v_ = *(float4*)(A + (((((int)blockIdx.x) * 2048) + (i * 8)) + (j * 4)));
        float4 v__1 = *(float4*)(B + (((((int)blockIdx.x) * 2048) + (i * 8)) + (j * 4)));
        __1.x = (v_.x+v__1.x);
        __1.y = (v_.y+v__1.y);
        __1.z = (v_.z+v__1.z);
        __1.w = (v_.w+v__1.w);
      *(float4*)(sum + (j * 4)) = __1;
    }
    if (((int)threadIdx.x) < 8) {
      #pragma unroll
      for (int i_1 = 0; i_1 < 1; ++i_1) {
        C[(((((int)blockIdx.x) * 2048) + (i * 8)) + ((int)threadIdx.x))] = sum[((int)threadIdx.x)];
      }
    }
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

extern "C" int call(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, musaStream_t stream=musaStreamDefault) {
        main_kernel<<<dim3(2, 1, 1), dim3(256, 1, 1), 0, stream>>>(A, B, C);
        TILELANG_CHECK_LAST_ERROR("main_kernel");

        return 0;
}
