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

extern "C" __global__ void gemm_ws_kernel(__attribute__((grid_constant)) const MUtensorDescriptor A_desc_0, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) gemm_ws_kernel(__attribute__((grid_constant)) const MUtensorDescriptor A_desc_0, const half_t* __restrict__ B, half_t* __restrict__ C) {
  int __partial_barrier_sync_phase_0 = 0;
  int __partial_barrier_sync_phase_1 = 0;
  __musa_async_bar_record(6);
  if (tl::tl_shuffle_elect<0>()) {
    __musa_async_init_arrival(5, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(6, ((128 + 31) / 32), 0);
  }
  __syncthreads_lm();
  extern __shared__ __align__(4096) uchar buf_dyn_shmem[];
  float C_local[128];
  if (tl::tl_shuffle_elect<0>()) {
    __musa_async_init_arrival(1, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(2, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(3, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(4, ((128 + 31) / 32), 0);
  }
  __syncthreads_lm();
  if ((128 <= ((int)threadIdx.x))) {
    __musa_async_arrive(5);
    __musa_async_wait(5, __partial_barrier_sync_phase_0);
    __partial_barrier_sync_phase_0 = (__partial_barrier_sync_phase_0 ^ 1);
    for (int ko = 0; ko < 32; ++ko) {
      __musa_async_wait(((ko & 1) + 3), (((ko & 3) >> 1) ^ 1));
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        tl::cp_async_gs<16>((&(((half_t*)buf_dyn_shmem)[(((((((ko & 1) * 4096) + (i * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) & 15) >> 1) ^ ((((int)threadIdx.x) & 127) >> 4)) * 16)) + ((((int)threadIdx.x) & 1) * 8)) - 1024)])), (&(B[((((((ko * 32768) + (i * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) - 8192)])));
      }
      tl::cp_async_commit();
      tl::cp_async_wait<0>();
      __musa_async_arrive(6);
      __musa_async_wait(6, __partial_barrier_sync_phase_1);
      __partial_barrier_sync_phase_1 = (__partial_barrier_sync_phase_1 ^ 1);
      if (tl::tl_shuffle_elect<128>()) {
        tl::mbarrier_arrive_expect_tx(((ko & 1) + 1), 8192);
        tl::tma_load<SmemSwizzleGranularity::B16, SmemSwizzleStride::B256, SmemSwizzleLine::B256, CacheHint::CACHE_NORMAL, CacheHint::CACHE_NORMAL>(A_desc_0, ((ko & 1) + 1), (&(((half_t*)buf_dyn_shmem)[(((ko & 1) * 4096) + 8192)])), (ko * 32), (((int)blockIdx.y) * 128), 32, 128);
      }
    }
  }
  if ((((int)threadIdx.x) < 128)) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 32; ++i_1) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(tl_f4*)(C_local + (i_1 * 4)) = (tl_f4{static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var), static_cast<float>(broadcast_var)});
    }
    for (int ko_1 = 0; ko_1 < 32; ++ko_1) {
      __musa_async_wait(((ko_1 & 1) + 1), ((ko_1 & 3) >> 1));
      tl::gemm_ss<128, 128, 32, 4, 1, 0, 0, 0, 32, 128, 0, 0, true, 128, 128, 32>((&(((half_t*)buf_dyn_shmem)[(((ko_1 & 1) * 4096) + 8192)])), (&(((half_t*)buf_dyn_shmem)[((ko_1 & 1) * 4096)])), (&(C_local[0])), 0);
      __musa_async_arrive(((ko_1 & 1) + 3));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 128; ++i_2) {
      C[((((((((int)blockIdx.y) * 131072) + ((i_2 >> 4) * 16384)) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.x) * 128)) + ((i_2 & 15) * 8)) + (((int)threadIdx.x) & 7))] = ((half_t)C_local[i_2]);
    }
  }
}
