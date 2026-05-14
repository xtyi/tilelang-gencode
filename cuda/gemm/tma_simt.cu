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

extern "C" __global__ void gemm_ws_simt_producer_kernel(__grid_constant__ const CUtensorMap A_desc, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) gemm_ws_simt_producer_kernel(__grid_constant__ const CUtensorMap A_desc, const half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t mbarrier_mem[4];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float C_local[128];
  half_t C_local_cast[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(128);
    mbarrier[1].init(128);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::__sync_thread_partial<3, 128>();
    for (int ko = 0; ko < 32; ++ko) {
      mbarrier[((ko & 1) + 2)].wait((((ko & 3) >> 1) ^ 1));
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        *(uint4*)(((half_t*)buf_dyn_shmem) + (((((((((ko & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 7680)) = *(uint4*)(B + ((((((ko * 32768) + (i * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) - 8192));
      }
      // to fix
      // tl::cp_async_commit();
      tl::__sync_thread_partial<3, 128>();
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(ko & 1)].expect_transaction(8192);
        tl::fence_proxy_async();
        tl::tma_load(A_desc, mbarrier[(ko & 1)], (&(((half_t*)buf_dyn_shmem)[((ko & 1) * 4096)])), (ko * 32), (((int)blockIdx.y) * 128));
      }
      tl::mbarrier_cp_async_arrive_noinc(mbarrier[(ko & 1)]);
    }
  } else {
    #pragma unroll
    for (int i_1 = 0; i_1 < 32; ++i_1) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_local + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
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
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          #pragma unroll
          for (int ki = 0; ki < 2; ++ki) {
            tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a + (((i_2 * 4096) + (ki * 32)) >> 4)), uint64_t(desc_b + ((ki * 2048) >> 4)), ((uint32_t*)(C_local + (i_2 * 64))), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(C_local + 0), 128);
      }
      mbarrier[((ko_1 & 1) + 2)].arrive();
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 64; ++i_3) {
      uint1 __1;
      float2 v_ = *(float2*)(C_local + (i_3 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(C_local_cast + 0) = __1;
      *(uint1*)(C + ((((((((((int)blockIdx.y) * 131072) + ((i_3 >> 5) * 65536)) + ((((int)threadIdx.x) >> 5) * 16384)) + ((i_3 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 128)) + (((i_3 & 31) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_local_cast + 0);
    }
  }
}
