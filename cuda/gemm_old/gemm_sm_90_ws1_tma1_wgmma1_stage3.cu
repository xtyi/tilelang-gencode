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

extern "C" __global__ void gemm_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) gemm_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[128];
  tl::GmmaDescriptor desc_a;
  tl::GmmaDescriptor desc_b;
  __shared__ uint64_t mbarrier_mem[6];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    mbarrier[0].init(128);
    mbarrier[1].init(128);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
    mbarrier[4].init(128);
    mbarrier[5].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 32; ++k) {
      mbarrier[((k % 3) + 3)].wait((((k % 6) / 3) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k % 3)].expect_transaction(8192);
        tl::fence_proxy_async();
        tl::tma_load(A_desc, mbarrier[(k % 3)], (&(((half_t*)buf_dyn_shmem)[((k % 3) * 4096)])), (k * 32), (((int)blockIdx.y) * 128));
        mbarrier[(k % 3)].expect_transaction(8192);
        tl::fence_proxy_async();
        tl::tma_load(B_desc, mbarrier[(k % 3)], (&(((half_t*)buf_dyn_shmem)[(((k % 3) * 4096) + 12288)])), (((int)blockIdx.x) * 128), (k * 32));
        tl::tma_load(B_desc, mbarrier[(k % 3)], (&(((half_t*)buf_dyn_shmem)[(((k % 3) * 4096) + 14336)])), ((((int)blockIdx.x) * 128) + 64), (k * 32));
      }
      mbarrier[(k % 3)].arrive();
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
      *(float2*)(C_local + (i * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    for (int k_1 = 0; k_1 < 32; ++k_1) {
      mbarrier[(k_1 % 3)].wait(((k_1 % 6) / 3));
      tl::initialize_wgmma_descriptor<2, 1, 32>(desc_a, (&(((half_t*)buf_dyn_shmem)[((k_1 % 3) * 4096)])));
      tl::initialize_wgmma_descriptor<1, 256, 64>(desc_b, (&(((half_t*)buf_dyn_shmem)[(((k_1 % 3) * 4096) + 12288)])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(C_local + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int i_1 = 0; i_1 < 2; ++i_1) {
        #pragma unroll
        for (int ki = 0; ki < 2; ++ki) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a + (((i_1 * 4096) + (ki * 32)) >> 4)), uint64_t(desc_b + ((ki * 2048) >> 4)), ((uint32_t*)(C_local + (i_1 * 64))), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(C_local + 0), 128);
      mbarrier[((k_1 % 3) + 3)].arrive();
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 64; ++i_2) {
      uint1 __1;
      float2 v_ = *(float2*)(C_local + (i_2 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(C + ((((((((((int)blockIdx.y) * 131072) + ((i_2 >> 5) * 65536)) + ((((int)threadIdx.x) >> 5) * 16384)) + ((i_2 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 128)) + (((i_2 & 31) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
    }
  }
}

