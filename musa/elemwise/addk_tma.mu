#include <tl_templates/musa/common.h>
#include <tl_templates/musa/intrin.h>
#include <tl_templates/musa/gemm.h>
#include <tl_templates/musa/copy.h>
#include <tl_templates/musa/barrier.h>
#include <tl_templates/musa/reduce.h>
#include <tl_templates/musa/threadblock_swizzle.h>
#include <tl_templates/musa/debug.h>

extern "C" __global__ void kernel_kernel(__attribute__((grid_constant)) const MUtensorDescriptor A_desc, __attribute__((grid_constant)) const MUtensorDescriptor C_desc);
extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel(__attribute__((grid_constant)) const MUtensorDescriptor A_desc, __attribute__((grid_constant)) const MUtensorDescriptor C_desc) {
  __musa_async_bar_record(2);
  extern __shared__ __align__(4096) float tile[];
  if (tl::tl_shuffle_elect<0>()) {
    __musa_async_init_arrival(2, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(1, ((32 + 31) / 32), 0);
  }
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    if (tl::tl_shuffle_elect<128>()) {
      __musa_async_add_trans(1, 65536);
      __musa_async_arrive(1);
      tl::tma_load<SmemSwizzleGranularity::NONE>(A_desc, 1, (&(tile[0])), (((int)blockIdx.x) * 128), (((int)blockIdx.y) * 128), 128, 128);
    }
  } else {
    __musa_async_wait(1, 0);
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      float4 __1;
        float4 v_ = *(float4*)(tile + ((i * 512) + (((int)threadIdx.x) * 4)));
        float4 v__1 = make_float4(0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/);
        __1.x = (v_.x+v__1.x);
        __1.y = (v_.y+v__1.y);
        __1.z = (v_.z+v__1.z);
        __1.w = (v_.w+v__1.w);
      *(float4*)(tile + ((i * 512) + (((int)threadIdx.x) * 4))) = __1;
    }
    __musa_async_arrive(2);
    __musa_async_wait(2, 0);
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store(C_desc, (&(tile[0])), (((int)blockIdx.x) * 128), (((int)blockIdx.y) * 128), 128, 128);
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

    musaError_t result_kernel_kernel = musaFuncSetAttribute(kernel_kernel, musaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    if (result_kernel_kernel != musaSuccess) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 65536, musaGetErrorString(result_kernel_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(float* __restrict__ A, float* __restrict__ C, musaStream_t stream=musaStreamDefault) {

        MUtensorDescriptor A_desc;
        MUtensorDescriptorDataType A_desc_type= (MUtensorDescriptorDataType)8;
        muuint32_t A_desc_tensorRank= 2;
        void *A_desc_globalAddress= A;
        muuint64_t A_desc_globalDim[2]= {static_cast<muuint64_t>(1024),static_cast<muuint64_t>(1024)};
        muuint64_t A_desc_globalStride[2]= {4,4096};
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

        MUtensorDescriptor C_desc;
        MUtensorDescriptorDataType C_desc_type= (MUtensorDescriptorDataType)8;
        muuint32_t C_desc_tensorRank= 2;
        void *C_desc_globalAddress= C;
        muuint64_t C_desc_globalDim[2]= {static_cast<muuint64_t>(1024),static_cast<muuint64_t>(1024)};
        muuint64_t C_desc_globalStride[2]= {4,4096};
        MUtensorDescriptorInterleave C_desc_interleave= (MUtensorDescriptorInterleave)0;
        muuint32_t C_desc_oobFill= 0;

        MUresult C_desc_result = muTensorDescriptorEncode(
    &C_desc, C_desc_type, C_desc_tensorRank, C_desc_globalAddress, C_desc_globalDim, C_desc_globalStride + 1, C_desc_interleave, C_desc_oobFill);

        if (C_desc_result != MUSA_SUCCESS) {
                std::stringstream ss;
                ss << "Error: Failed to initialize the TMA descriptor C_desc";
                snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
                return -1;
        }
        kernel_kernel<<<dim3(8, 8, 1), dim3(256, 1, 1), 65536, stream>>>(A_desc, C_desc);
        TILELANG_CHECK_LAST_ERROR("kernel_kernel");

        return 0;
}
