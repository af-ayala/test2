#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

using float32_t = float;
using int32x4_t __attribute__((ext_vector_type(4))) = int;
using float32x4_t __attribute__((ext_vector_type(4))) = float;

__device__ float32_t
    llvm_amdgcn_raw_buffer_load_f32(int32x4_t srsrc,
                                    uint32_t  voffset,
                                    uint32_t  soffset,
                                    int32_t   glc) __asm("llvm.amdgcn.raw.buffer.load.f32");

__device__ void
    llvm_amdgcn_raw_buffer_store_f32(float32_t data,
                                     int32x4_t srsrc,
                                     uint32_t  voffset,
                                     uint32_t  soffset,
                                     int32_t   glc) __asm("llvm.amdgcn.raw.buffer.store.f32");

struct alignas(16) BufferResource
{
    union Desc {
        int32x4_t d128;
        void*     d64[2];
        uint32_t  d32[4];
    };

    __forceinline__ __device__
    BufferResource(void const* base_addr, uint32_t num_records = 0xFFFFFFFE)
    {
        desc_.d64[0] = const_cast<void*>(base_addr);
        desc_.d32[2] = num_records;
        desc_.d32[3] = 0x00020000;  // CDNA buffer resource descriptor
    }

    __forceinline__ __device__
    operator int32x4_t()
    {
        Desc ret;
        ret.d32[0] = __builtin_amdgcn_readfirstlane(desc_.d32[0]);
        ret.d32[1] = __builtin_amdgcn_readfirstlane(desc_.d32[1]);
        ret.d64[1] = desc_.d64[1];
        return ret.d128;
    }
    Desc desc_;
};

// Kernel using buffer intrinsics (like rocFFT does)
__global__ void saxpy_buffer_intrinsic(float* y, const float* x, float a, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        BufferResource rsc_x(x);
        BufferResource rsc_y(y);
        float xval = llvm_amdgcn_raw_buffer_load_f32(rsc_x, i * sizeof(float), 0, 0);
        float yval = llvm_amdgcn_raw_buffer_load_f32(rsc_y, i * sizeof(float), 0, 0);
        float result = a * xval + yval;
        llvm_amdgcn_raw_buffer_store_f32(result, rsc_y, i * sizeof(float), 0, 0);
    }
}

// Kernel using normal global load/store (our previous test)
__global__ void saxpy_normal(float* y, const float* x, float a, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = a * x[i] + y[i];
}

void run_test(const char* label, void(*kernel)(float*, const float*, float, int), int N, float a)
{
    float *d_x, *d_y;
    float* h_x = new float[N];
    float* h_y = new float[N];

    for (int i = 0; i < N; i++) { h_x[i] = 1.0f; h_y[i] = 2.0f; }

    hipMalloc(&d_x, N * sizeof(float));
    hipMalloc(&d_y, N * sizeof(float));
    hipMemcpy(d_x, h_x, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, h_y, N * sizeof(float), hipMemcpyHostToDevice);

    kernel<<<(N+255)/256, 256>>>(d_y, d_x, a, N);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("  %-25s  KERNEL ERROR: %s\n", label, hipGetErrorString(err));
        hipFree(d_x); hipFree(d_y); delete[] h_x; delete[] h_y;
        return;
    }

    hipMemcpy(h_y, d_y, N * sizeof(float), hipMemcpyDeviceToHost);

    double maxErr = 0;
    for (int i = 0; i < N; i++)
        maxErr = fmax(maxErr, fabs(h_y[i] - 5.0f));

    printf("  %-25s  maxErr=%e  %s\n", label, maxErr, maxErr < 1e-6 ? "PASS" : "FAIL");

    hipFree(d_x); hipFree(d_y); delete[] h_x; delete[] h_y;
}

int main()
{
    const char* xnack = getenv("HSA_XNACK");
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("HSA_XNACK=%s  arch=%s\n", xnack ? xnack : "unset", prop.gcnArchName);

    const int N = 1 << 20;
    run_test("normal (flat_load)", saxpy_normal, N, 3.0f);
    run_test("buffer_intrinsic", saxpy_buffer_intrinsic, N, 3.0f);
}
