#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void gemm_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) gemm_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[128];
  tl::GmmaDescriptor desc_a;
  tl::GmmaDescriptor desc_b;
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    *(float2*)(C_local + (i * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((i_1 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+((((((int)blockIdx.y) * 131072) + (i_1 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), B+((((i_2 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 31; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((k + 1) & 1) * 8192) + (i_3 * 2048)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+((((((((int)blockIdx.y) * 131072) + (i_3 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::cp_async_gs<16>(buf_dyn_shmem+((((((((((k + 1) & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), B+((((((k * 32768) + (i_4 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 32768));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads();
    tl::initialize_wgmma_descriptor<2, 1, 32>(desc_a, (&(((half_t*)buf_dyn_shmem)[((k & 1) * 4096)])));
    tl::initialize_wgmma_descriptor<1, 256, 64>(desc_b, (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 4096) + 8192)])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(C_local + 0), 128);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      #pragma unroll
      for (int ki = 0; ki < 2; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a + (((i_5 * 4096) + (ki * 32)) >> 4)), uint64_t(desc_b + ((ki * 2048) >> 4)), ((uint32_t*)(C_local + (i_5 * 64))), 1);
      }
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(C_local + 0), 128);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::initialize_wgmma_descriptor<2, 1, 32>(desc_a, (&(((half_t*)buf_dyn_shmem)[4096])));
  tl::initialize_wgmma_descriptor<1, 256, 64>(desc_b, (&(((half_t*)buf_dyn_shmem)[12288])));
  tl::warpgroup_fence_operand(reinterpret_cast<float*>(C_local + 0), 128);
  tl::warpgroup_arrive();
  tl::fence_proxy_async();
  #pragma unroll
  for (int i_6 = 0; i_6 < 2; ++i_6) {
    #pragma unroll
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a + (((i_6 * 4096) + (ki_1 * 32)) >> 4)), uint64_t(desc_b + ((ki_1 * 2048) >> 4)), ((uint32_t*)(C_local + (i_6 * 64))), 1);
    }
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_wait<0>();
  tl::warpgroup_fence_operand(reinterpret_cast<float*>(C_local + 0), 128);
  #pragma unroll
  for (int i_7 = 0; i_7 < 64; ++i_7) {
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_7 * 2));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    *(uint1*)(C + ((((((((((int)blockIdx.y) * 131072) + ((i_7 >> 5) * 65536)) + ((((int)threadIdx.x) >> 5) * 16384)) + ((i_7 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 128)) + (((i_7 & 31) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
  }
}

