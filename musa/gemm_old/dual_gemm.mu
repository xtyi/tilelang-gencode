#include <tl_templates/musa/common.h>
#include <tl_templates/musa/intrin.h>
#include <tl_templates/musa/gemm.h>
#include <tl_templates/musa/copy.h>
#include <tl_templates/musa/barrier.h>
#include <tl_templates/musa/reduce.h>
#include <tl_templates/musa/threadblock_swizzle.h>
#include <tl_templates/musa/debug.h>

extern "C" __global__ void main_kernel(__attribute__((grid_constant)) const MUtensorDescriptor K_desc, half_t* __restrict__ Output, __attribute__((grid_constant)) const MUtensorDescriptor Q_desc, __attribute__((grid_constant)) const MUtensorDescriptor V_desc);
extern "C" __global__ void __launch_bounds__(640, 1) main_kernel(__attribute__((grid_constant)) const MUtensorDescriptor K_desc, half_t* __restrict__ Output, __attribute__((grid_constant)) const MUtensorDescriptor Q_desc, __attribute__((grid_constant)) const MUtensorDescriptor V_desc) {
  int __musa_sync_phase_0 = 0;
  int __musa_sync_phase_1 = 0;
  __musa_async_bar_record(11);
  extern __shared__ __align__(4096) uchar buf_dyn_shmem[];
  float acc_o[64];
  float acc_s[32];
  if (tl::tl_shuffle_elect<0>()) {
    __musa_async_init_arrival(10, ((512 + 31) / 32), 0);
    __musa_async_init_arrival(11, ((512 + 31) / 32), 0);
    __musa_async_init_arrival(1, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(2, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(3, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(4, ((32 + 31) / 32), 0);
    __musa_async_init_arrival(5, ((512 + 31) / 32), 0);
    __musa_async_init_arrival(6, ((512 + 31) / 32), 0);
    __musa_async_init_arrival(7, ((512 + 31) / 32), 0);
    __musa_async_init_arrival(8, ((512 + 31) / 32), 0);
    __musa_async_init_arrival(9, ((32 + 31) / 32), 0);
  }
  __syncthreads();
  if (512 <= ((int)threadIdx.x)) {
    if (tl::tl_shuffle_elect<128>()) {
      __musa_async_add_trans(9, 65536);
      __musa_async_arrive(9);
      tl::tma_load<SmemSwizzleGranularity::B16>(Q_desc, 9, (&(((half_t*)buf_dyn_shmem)[0])), 0, 0, 128, 256);
    }
    if (tl::tl_shuffle_elect<128>()) {
      for (int k = 0; k < 2; ++k) {
        if (tl::tl_shuffle_elect<128>()) {
          __musa_async_wait((k + 5), (0 ^ 1));
          __musa_async_add_trans((k + 1), 16384);
          __musa_async_arrive((k + 1));
          tl::tma_load<SmemSwizzleGranularity::B16>(K_desc, (k + 1), (&(((half_t*)buf_dyn_shmem)[((k * 8192) + 32768)])), 0, (k * 64), 128, 64);
          __musa_async_wait((k + 7), (0 ^ 1));
          __musa_async_add_trans((k + 3), 16384);
          __musa_async_arrive((k + 3));
          tl::tma_load<SmemSwizzleGranularity::B32>(V_desc, (k + 3), (&(((half_t*)buf_dyn_shmem)[((k * 8192) + 49152)])), 0, (k * 64), 64, 64);
          tl::tma_load<SmemSwizzleGranularity::B32>(V_desc, (k + 3), (&(((half_t*)buf_dyn_shmem)[((k * 8192) + 53248)])), 64, (k * 64), 64, 64);
        }
      }
    }
  } else {
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
      acc_o[i] = 0x0p+0f/*0.000000e+00*/;
    }
    __musa_async_wait(9, 0);
    for (int k_1 = 0; k_1 < 2; ++k_1) {
      #pragma unroll
      for (int i_1 = 0; i_1 < 32; ++i_1) {
        acc_s[i_1] = 0x0p+0f/*0.000000e+00*/;
      }
      __musa_async_wait((k_1 + 1), 0);
      tl::gemm_ss<256, 64, 128, 8, 2, 0, 1, 0, 128, 128, 0, 0, true>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[((k_1 * 8192) + 32768)])), (&(acc_s[0])));
      __musa_sqmma_wait();
      __musa_async_arrive((k_1 + 5));
      __musa_async_arrive(10);
      __musa_async_wait(10, __musa_sync_phase_0);
      __musa_sync_phase_0 = (__musa_sync_phase_0 ^ 1);
      #pragma unroll
      for (int i_2 = 0; i_2 < 32; ++i_2) {
        ((half_t*)buf_dyn_shmem)[((((((((((int64_t)((int)threadIdx.x)) & (int64_t)255) >> (int64_t)7) * (int64_t)8192) + ((((int64_t)i_2) >> (int64_t)2) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)4) * (int64_t)128)) + (((((((((int64_t)((int)threadIdx.x)) & (int64_t)15) >> (int64_t)3) * (int64_t)8) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)8) * (int64_t)4)) + (((int64_t)i_2) & (int64_t)3)) ^ ((((((int64_t)i_2) & (int64_t)7) >> (int64_t)2) * (int64_t)8) + ((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)4))) * (int64_t)8)) + (((int64_t)((int)threadIdx.x)) & (int64_t)7)) + (int64_t)65536)] = ((half_t)acc_s[i_2]);
      }
      __musa_async_wait((k_1 + 3), 0);
      __musa_async_arrive(11);
      __musa_async_wait(11, __musa_sync_phase_1);
      __musa_sync_phase_1 = (__musa_sync_phase_1 ^ 1);
      tl::gemm_ss<256, 128, 64, 8, 2, 0, 0, 0, 64, 128, 0, 0, true>((&(((half_t*)buf_dyn_shmem)[65536])), (&(((half_t*)buf_dyn_shmem)[((k_1 * 8192) + 49152)])), (&(acc_o[0])));
      __musa_sqmma_wait();
      __musa_async_arrive((k_1 + 7));
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 64; ++i_3) {
      Output[((((((((((int)threadIdx.x) & 255) >> 7) * 16384) + ((i_3 >> 3) * 2048)) + (((((int)threadIdx.x) & 127) >> 3) * 128)) + ((((int)threadIdx.x) >> 8) * 64)) + ((i_3 & 7) * 8)) + (((int)threadIdx.x) & 7))] = ((half_t)acc_o[i_3]);
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

    musaError_t result_main_kernel = musaFuncSetAttribute(main_kernel, musaFuncAttributeMaxDynamicSharedMemorySize, 163840);
    if (result_main_kernel != musaSuccess) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 163840, musaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ Q, half_t* __restrict__ K, half_t* __restrict__ V, half_t* __restrict__ Output, musaStream_t stream=musaStreamDefault) {

        MUtensorDescriptor K_desc;
        MUtensorDescriptorDataType K_desc_type= (MUtensorDescriptorDataType)4;
        muuint32_t K_desc_tensorRank= 2;
        void *K_desc_globalAddress= K;
        muuint64_t K_desc_globalDim[2]= {static_cast<muuint64_t>(128),static_cast<muuint64_t>(128)};
        muuint64_t K_desc_globalStride[2]= {2,256};
        MUtensorDescriptorInterleave K_desc_interleave= (MUtensorDescriptorInterleave)0;
        muuint32_t K_desc_oobFill= 0;

        MUresult K_desc_result = muTensorDescriptorEncode(
    &K_desc, K_desc_type, K_desc_tensorRank, K_desc_globalAddress, K_desc_globalDim, K_desc_globalStride + 1, K_desc_interleave, K_desc_oobFill);

        if (K_desc_result != MUSA_SUCCESS) {
                std::stringstream ss;
                ss << "Error: Failed to initialize the TMA descriptor K_desc";
                snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
                return -1;
        }

        MUtensorDescriptor Q_desc;
        MUtensorDescriptorDataType Q_desc_type= (MUtensorDescriptorDataType)4;
        muuint32_t Q_desc_tensorRank= 2;
        void *Q_desc_globalAddress= Q;
        muuint64_t Q_desc_globalDim[2]= {static_cast<muuint64_t>(128),static_cast<muuint64_t>(256)};
        muuint64_t Q_desc_globalStride[2]= {2,256};
        MUtensorDescriptorInterleave Q_desc_interleave= (MUtensorDescriptorInterleave)0;
        muuint32_t Q_desc_oobFill= 0;

        MUresult Q_desc_result = muTensorDescriptorEncode(
    &Q_desc, Q_desc_type, Q_desc_tensorRank, Q_desc_globalAddress, Q_desc_globalDim, Q_desc_globalStride + 1, Q_desc_interleave, Q_desc_oobFill);

        if (Q_desc_result != MUSA_SUCCESS) {
                std::stringstream ss;
                ss << "Error: Failed to initialize the TMA descriptor Q_desc";
                snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
                return -1;
        }

        MUtensorDescriptor V_desc;
        MUtensorDescriptorDataType V_desc_type= (MUtensorDescriptorDataType)4;
        muuint32_t V_desc_tensorRank= 2;
        void *V_desc_globalAddress= V;
        muuint64_t V_desc_globalDim[2]= {static_cast<muuint64_t>(128),static_cast<muuint64_t>(128)};
        muuint64_t V_desc_globalStride[2]= {2,256};
        MUtensorDescriptorInterleave V_desc_interleave= (MUtensorDescriptorInterleave)0;
        muuint32_t V_desc_oobFill= 0;

        MUresult V_desc_result = muTensorDescriptorEncode(
    &V_desc, V_desc_type, V_desc_tensorRank, V_desc_globalAddress, V_desc_globalDim, V_desc_globalStride + 1, V_desc_interleave, V_desc_oobFill);

        if (V_desc_result != MUSA_SUCCESS) {
                std::stringstream ss;
                ss << "Error: Failed to initialize the TMA descriptor V_desc";
                snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
                return -1;
        }
        main_kernel<<<dim3(1, 1, 1), dim3(640, 1, 1), 163840, stream>>>(K_desc, Output, Q_desc, V_desc);
        TILELANG_CHECK_LAST_ERROR("main_kernel");

        return 0;
}
