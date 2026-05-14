#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void tma_load_add_one_kernel(const float* __restrict__ A, __grid_constant__ const CUtensorMap C_desc);
extern "C" __global__ void __launch_bounds__(128, 1) tma_load_add_one_kernel(const float* __restrict__ A, __grid_constant__ const CUtensorMap C_desc) {
  extern __shared__ __align__(1024) float tile[];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(C_desc);
  }
  tile[(((((((int)threadIdx.x) >> 4) * 16) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = A[((((((int)blockIdx.y) * 1024) + ((((int)threadIdx.x) >> 4) * 128)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) & 15))];
  tile[(((((((int)threadIdx.x) >> 4) * 16) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = (tile[(((((((int)threadIdx.x) >> 4) * 16) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3))] + 0x1p+0f/*1.000000e+00*/);
  __syncthreads();
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    tl::tma_store(C_desc, (&(tile[0])), (((int)blockIdx.x) * 16), (((int)blockIdx.y) * 8));
    tl::tma_store_arrive();
    tl::tma_store_wait<0>();
  }
}
