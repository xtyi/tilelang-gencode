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

extern "C" __global__ void gemm_ws_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc);
extern "C" __global__ void __launch_bounds__(256, 1) gemm_ws_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t mbarrier_mem[4];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float C_local[128];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::prefetch_tma_descriptor(C_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int ko = 0; ko < 32; ++ko) {
      mbarrier[((ko & 1) + 2)].wait((((ko & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(ko & 1)].expect_transaction(8192);
        tl::tma_load(A_desc, mbarrier[(ko & 1)], (&(((half_t*)buf_dyn_shmem)[((ko & 1) * 4096)])), (ko * 32), (((int)blockIdx.y) * 128));
        mbarrier[(ko & 1)].arrive_and_expect_tx(8192);
        tl::tma_load(B_desc, mbarrier[(ko & 1)], (&(((half_t*)buf_dyn_shmem)[(((ko & 1) * 4096) + 8192)])), (((int)blockIdx.x) * 128), (ko * 32));
        tl::tma_load(B_desc, mbarrier[(ko & 1)], (&(((half_t*)buf_dyn_shmem)[(((ko & 1) * 4096) + 10240)])), ((((int)blockIdx.x) * 128) + 64), (ko * 32));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int ko_1 = 0; ko_1 < 32; ++ko_1) {
      mbarrier[(ko_1 & 1)].wait(((ko_1 & 3) >> 1));
      {
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<2, 1, 32>(desc_a, (&(((half_t*)buf_dyn_shmem)[((ko_1 & 1) * 4096)])));
        tl::initialize_wgmma_descriptor<1, 256, 64>(desc_b, (&(((half_t*)buf_dyn_shmem)[(((ko_1 & 1) * 4096) + 8192)])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(C_local + 0), 128);
        tl::warpgroup_arrive();
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
      }
      mbarrier[((ko_1 & 1) + 2)].arrive();
    }
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      tl::ptx_stmatrix_x4((&(((half_t*)buf_dyn_shmem)[(((((((i_2 & 7) >> 2) * 8192) + ((i_2 >> 3) * 4096)) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)C_local[(i_2 * 8)]), ((half_t)C_local[((i_2 * 8) + 1)])), __pack_half2(((half_t)C_local[((i_2 * 8) + 2)]), ((half_t)C_local[((i_2 * 8) + 3)])), __pack_half2(((half_t)C_local[((i_2 * 8) + 4)]), ((half_t)C_local[((i_2 * 8) + 5)])), __pack_half2(((half_t)C_local[((i_2 * 8) + 6)]), ((half_t)C_local[((i_2 * 8) + 7)])));
    }
    tl::__sync_thread_partial<3, 128>();
    if (tl::tl_shuffle_elect<128>()) {
      tl::fence_proxy_async();
      tl::tma_store(C_desc, (&(((half_t*)buf_dyn_shmem)[0])), (((int)blockIdx.x) * 128), (((int)blockIdx.y) * 128));
      tl::tma_store(C_desc, (&(((half_t*)buf_dyn_shmem)[8192])), ((((int)blockIdx.x) * 128) + 64), (((int)blockIdx.y) * 128));
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
    }
  }
}
