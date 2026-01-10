#include <tl_templates/cuda/instruction/mma.h>
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
  half_t A_local[32];
  half_t B_local[32];
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    *(float2*)(C_local + (i * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }

  // prologue: load A for Stage1
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((i_1 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+((((((int)blockIdx.y) * 131072) + (i_1 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)));
  }

  // prologue: load B for Stage1
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), B+((((i_2 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }

  // prologue: commit for Stage1
  tl::cp_async_commit();


  // prologue: loadA for Stage2
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((i_3 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 8192), A+(((((((int)blockIdx.y) * 131072) + (i_3 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)) + 32));
  }

  // prologue: loadB for Stage2
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), B+(((((i_4 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 32768));
  }

  // prologue: commit for Stage2
  tl::cp_async_commit();


  // Steady State
  for (int k = 0; k < 30; ++k) {
    __syncthreads();

    // load A
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((k + 2) % 3) * 8192) + (i_5 * 2048)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+((((((((int)blockIdx.y) * 131072) + (i_5 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64));
    }
    // load B
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs<16>(buf_dyn_shmem+((((((((((k + 2) % 3) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_6 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), B+((((((k * 32768) + (i_6 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 65536));
    }
    // commit
    tl::cp_async_commit();

    // wait in-flight number'max is 2
    tl::cp_async_wait<2>();

    __syncthreads();

    // gemm
    for (int ki = 0; ki < 2; ++ki) {
      for (int i_7 = 0; i_7 < 4; ++i_7) {
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[(((((((k % 3) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_7 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])) + 0, A_local + (i_7 * 8));
      }
      for (int i_8 = 0; i_8 < 4; ++i_8) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((((((k % 3) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_8 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])) + 0, B_local + (i_8 * 8));
      }
      for (int i_9 = 0; i_9 < 4; ++i_9) {
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_9 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_9 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_9 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_9 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
  }


  // epilogue: wait
  tl::cp_async_wait<1>();

  __syncthreads();

  // epilogue: gemm
  for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_10 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])) + 0, A_local + (i_10 * 8));
    }
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])) + 0, B_local + (i_11 * 8));
    }
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      for (int j_1 = 0; j_1 < 4; ++j_1) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_12 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local + (j_1 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_12 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 8) + 4)));
      }
    }
  }


  // epilogue: wait
  tl::cp_async_wait<0>();

  __syncthreads();

  // epilogue: gemm
  for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_13 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 4096)])) + 0, A_local + (i_13 * 8));
    }
    for (int i_14 = 0; i_14 < 4; ++i_14) {
      tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_14 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_14 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])) + 0, B_local + (i_14 * 8));
    }
    for (int i_15 = 0; i_15 < 4; ++i_15) {
      for (int j_2 = 0; j_2 < 4; ++j_2) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_15 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local + (j_2 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_15 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j_2 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 64; ++i_16) {
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_16 * 2));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    *(uint1*)(C + (((((((((((int)blockIdx.y) * 131072) + (((((int)threadIdx.x) & 63) >> 5) * 65536)) + ((i_16 >> 4) * 16384)) + ((i_16 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_16 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
  }
}

