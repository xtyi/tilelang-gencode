#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/fop.h>
#include <tl_templates/musa/accelerated_ops.h>
#include <tl_templates/musa/intrin.h>
#include <tl_templates/musa/gemm.h>
#include <tl_templates/musa/copy.h>
#include <tl_templates/musa/barrier.h>
#include <tl_templates/musa/reduce.h>
#include <tl_templates/musa/threadblock_swizzle.h>
#include <tl_templates/musa/debug.h>

extern "C" __global__ void gemm_ws_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) gemm_ws_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  __musa_async_bar_record(0);
  extern __shared__ __align__(4096) uchar buf_dyn_shmem[];
  float C_local[128];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(tl_f4*)(C_local + (i * 4)) = (tl_f4{static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var)});
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)buf_dyn_shmem)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((i_1 & 1) * 8) + (((int)threadIdx.x) >> 4))) * 8)) + 8192)])), (&(A[((((((int)blockIdx.y) * 131072) + (i_1 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8))])));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)buf_dyn_shmem)[((((i_2 * 1024) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) & 15) >> 1) ^ (((int)threadIdx.x) >> 4)) * 16)) + ((((int)threadIdx.x) & 1) * 8))])), (&(B[((((i_2 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  for (int ko = 0; ko < 31; ++ko) {
    __syncthreads_lm();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      tl::cp_async_gs<16>((&(((half_t*)buf_dyn_shmem)[(((((((ko + 1) & 1) * 4096) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((i_3 & 1) * 8) + (((int)threadIdx.x) >> 4))) * 8)) + 8192)])), (&(A[((((((((int)blockIdx.y) * 131072) + (i_3 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + (ko * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::cp_async_gs<16>((&(((half_t*)buf_dyn_shmem)[(((((((ko + 1) & 1) * 4096) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) & 15) >> 1) ^ (((int)threadIdx.x) >> 4)) * 16)) + ((((int)threadIdx.x) & 1) * 8))])), (&(B[((((((ko * 32768) + (i_4 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 32768)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads_lm();
    tl::gemm_ss<128, 128, 32, 4, 1, 0, 0, 0, 32, 128, 0, 0, true, 128, 128, 32>((&(((half_t*)buf_dyn_shmem)[(((ko & 1) * 4096) + 8192)])), (&(((half_t*)buf_dyn_shmem)[((ko & 1) * 4096)])), (&(C_local[0])), 0);
  }
  tl::cp_async_wait<0>();
  __syncthreads_lm();
  tl::gemm_ss<128, 128, 32, 4, 1, 0, 0, 0, 32, 128, 0, 0, true, 128, 128, 32>((&(((half_t*)buf_dyn_shmem)[12288])), (&(((half_t*)buf_dyn_shmem)[4096])), (&(C_local[0])), 0);
  #pragma unroll
  for (int i_5 = 0; i_5 < 128; ++i_5) {
    C[((((((((int)blockIdx.y) * 131072) + ((i_5 >> 4) * 16384)) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.x) * 128)) + ((i_5 & 15) * 8)) + (((int)threadIdx.x) & 7))] = ((half_t)C_local[i_5]);
  }
}
