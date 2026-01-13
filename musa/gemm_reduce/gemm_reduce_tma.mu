#include <tl_templates/musa/common.h>
#include <tl_templates/musa/intrin.h>
#include <tl_templates/musa/gemm.h>
#include <tl_templates/musa/copy.h>
#include <tl_templates/musa/barrier.h>
#include <tl_templates/musa/reduce.h>
#include <tl_templates/musa/threadblock_swizzle.h>
#include <tl_templates/musa/debug.h>

extern "C" __global__ void main_kernel(__attribute__((grid_constant)) const MUtensorDescriptor A_desc, __attribute__((grid_constant)) const MUtensorDescriptor B_desc, float* __restrict__ Out);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__attribute__((grid_constant)) const MUtensorDescriptor A_desc, __attribute__((grid_constant)) const MUtensorDescriptor B_desc, float* __restrict__ Out) {
  __musa_async_bar_record(2);
  extern __shared__ __align__(4096) uchar buf_dyn_shmem[];
  float acc[8];
  float sum[2];
  if (tl::tl_shuffle_elect<0>()) {
    __musa_async_init_arrival(2, ((256 + 31) / 32), 0);
    __musa_async_init_arrival(1, ((32 + 31) / 32), 0);
  }
  __syncthreads();
  if (256 <= ((int)threadIdx.x)) {
    if (tl::tl_shuffle_elect<128>()) {
      __musa_async_add_trans(1, 4096);
      tl::tma_load<SmemSwizzleGranularity::B16>(A_desc, 1, (&(((half_t*)buf_dyn_shmem)[4096])), 0, 0, 64, 32);
      __musa_async_add_trans(1, 8192);
      tl::tma_load<SmemSwizzleGranularity::B16>(B_desc, 1, (&(((half_t*)buf_dyn_shmem)[0])), 0, 0, 64, 64);
      __musa_async_arrive(1);
    }
  } else {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[i] = 0x0p+0f/*0.000000e+00*/;
    }
    __musa_async_wait(1, 0);
    tl::gemm_ss<32, 64, 64, 4, 2, 0, 1, 0, 64, 64, 0, 0, true>((&(((half_t*)buf_dyn_shmem)[4096])), (&(((half_t*)buf_dyn_shmem)[0])), (&(acc[0])));
    __musa_sqmma_wait();
    __musa_async_arrive(2);
    __musa_async_wait(2, 0);
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      sum[i_1] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv = 0; rv < 4; ++rv) {
        sum[i_1] = (sum[i_1] + acc[((i_1 * 4) + rv)]);
      }
      sum[i_1] = tl::AllReduce<tl::SumOp, 256, 128, 0>::run(sum[i_1], (&(((float*)buf_dyn_shmem)[0])));
      sum[i_1] = tl::AllReduce<tl::SumOp, 8, 1, 0>::run(sum[i_1]);
    }
    if ((((((int)threadIdx.x) & 7) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        Out[((i_2 * 16) + ((((int)threadIdx.x) & 127) >> 3))] = sum[i_2];
      }
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

    musaError_t result_main_kernel = musaFuncSetAttribute(main_kernel, musaFuncAttributeMaxDynamicSharedMemorySize, 12288);
    if (result_main_kernel != musaSuccess) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 12288, musaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ A, half_t* __restrict__ B, float* __restrict__ Out, musaStream_t stream=musaStreamDefault) {

        MUtensorDescriptor A_desc;
        MUtensorDescriptorDataType A_desc_type= (MUtensorDescriptorDataType)4;
        muuint32_t A_desc_tensorRank= 2;
        void *A_desc_globalAddress= A;
        muuint64_t A_desc_globalDim[2]= {static_cast<muuint64_t>(64),static_cast<muuint64_t>(32)};
        muuint64_t A_desc_globalStride[2]= {2,128};
        MUtensorDescriptorInterleave A_desc_interleave= (MUtensorDescriptorInterleave)0;
        muuint32_t A_desc_oobFill= 0;

        MUresult A_desc_result = muTensorDescriptorEncode(
    &A_desc, A_desc_type, A_desc_tensorRank, A_desc_globalAddress, A_desc_globalDim, A_desc_globalStride + 1, A_desc_interleave, A_desc_oobFill);

        if (A_desc_result != MUSA_SUCCESS) {
                std::stringstream ss;
                ss << "Error: Failed to initialize the TMA descriptor A_desc";
                snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
                return -1;
        }

        MUtensorDescriptor B_desc;
        MUtensorDescriptorDataType B_desc_type= (MUtensorDescriptorDataType)4;
        muuint32_t B_desc_tensorRank= 2;
        void *B_desc_globalAddress= B;
        muuint64_t B_desc_globalDim[2]= {static_cast<muuint64_t>(64),static_cast<muuint64_t>(64)};
        muuint64_t B_desc_globalStride[2]= {2,128};
        MUtensorDescriptorInterleave B_desc_interleave= (MUtensorDescriptorInterleave)0;
        muuint32_t B_desc_oobFill= 0;

        MUresult B_desc_result = muTensorDescriptorEncode(
    &B_desc, B_desc_type, B_desc_tensorRank, B_desc_globalAddress, B_desc_globalDim, B_desc_globalStride + 1, B_desc_interleave, B_desc_oobFill);

        if (B_desc_result != MUSA_SUCCESS) {
                std::stringstream ss;
                ss << "Error: Failed to initialize the TMA descriptor B_desc";
                snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
                return -1;
        }
        main_kernel<<<dim3(1, 1, 1), dim3(384, 1, 1), 12288, stream>>>(A_desc, B_desc, Out);
        TILELANG_CHECK_LAST_ERROR("main_kernel");

        return 0;
}
