#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void tma_load_add_one_ws_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap C_desc);
extern "C" __global__ void __launch_bounds__(256, 1) tma_load_add_one_ws_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap C_desc) {
  __shared__ __align__(16) uint64_t mbarrier_mem[4];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  extern __shared__ __align__(1024) half_t tile[];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
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
    for (int ko = 0; ko < 2; ++ko) {
      mbarrier[(ko + 2)].wait(1);
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[ko].arrive_and_expect_tx(1024);
        tl::tma_load(A_desc, mbarrier[ko], (&(tile[(ko * 512)])), (ko * 128), (((int)blockIdx.x) * 4));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    for (int ko_1 = 0; ko_1 < 2; ++ko_1) {
      mbarrier[ko_1].wait(0);
      tl::__sync_thread_partial<3, 128>();
      half_t broadcast_var = half_t(0x1p+0f/*1.000000e+00*/);
      uint2 __1;
        uint2 v_ = *(uint2*)(tile + ((ko_1 * 512) + (((int)threadIdx.x) * 4)));
        uint2 v__1 = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
        *(uint1*)(&(__1.x)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.x)))));
        *(uint1*)(&(__1.y)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.y))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.y)))));
      *(uint2*)(tile + ((ko_1 * 512) + (((int)threadIdx.x) * 4))) = __1;
      tl::__sync_thread_partial<3, 128>();
      if (tl::tl_shuffle_elect<128>()) {
        tl::fence_proxy_async();
        tl::tma_store(C_desc, (&(tile[(ko_1 * 512)])), (ko_1 * 128), (((int)blockIdx.x) * 4));
        tl::tma_store_arrive();
        tl::tma_store_wait<0>();
      }
      mbarrier[(ko_1 + 2)].arrive();
    }
  }
}
