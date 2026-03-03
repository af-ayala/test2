#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <cstdio>
#include <cmath>

#define HIP_CHECK(x) do { hipError_t e = (x); if(e) { fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1); } } while(0)
#define RTC_CHECK(x) do { hiprtcResult r = (x); if(r) { fprintf(stderr,"HIPRTC %s:%d %s\n",__FILE__,__LINE__,hiprtcGetErrorString(r)); exit(1); } } while(0)

// Kernel source with buffer intrinsics, matching what rocFFT generates
const char* kernel_src_buffer = R"(
using float32_t = float;
using int32x4_t __attribute__((ext_vector_type(4))) = int;

extern "C" __device__ float32_t
    llvm_amdgcn_raw_buffer_load_f32(int32x4_t srsrc,
                                    unsigned int voffset,
                                    unsigned int soffset,
                                    int glc) __asm("llvm.amdgcn.raw.buffer.load.f32");

extern "C" __device__ void
    llvm_amdgcn_raw_buffer_store_f32(float32_t data,
                                     int32x4_t srsrc,
                                     unsigned int voffset,
                                     unsigned int soffset,
                                     int glc) __asm("llvm.amdgcn.raw.buffer.store.f32");

struct alignas(16) BufferResource
{
    union Desc {
        int32x4_t d128;
        void*     d64[2];
        unsigned int d32[4];
    };

    __forceinline__ __device__
    BufferResource(void const* base_addr, unsigned int num_records = 0xFFFFFFFEu)
    {
        desc_.d64[0] = const_cast<void*>(base_addr);
        desc_.d32[2] = num_records;
        desc_.d32[3] = 0x00020000;
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

extern "C" __global__ void saxpy_buffer(float* y, const float* x, float a, int N)
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
)";

// Simple kernel without intrinsics
const char* kernel_src_normal = R"(
extern "C" __global__ void saxpy_normal(float* y, const float* x, float a, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = a * x[i] + y[i];
}
)";

bool rtc_compile_and_run(const char* label, const char* src, const char* func_name,
                         const char* arch_flag, int N, float alpha)
{
    hiprtcProgram prog;
    RTC_CHECK(hiprtcCreateProgram(&prog, src, "test.cu", 0, nullptr, nullptr));

    std::string arch_opt = std::string("--offload-arch=") + arch_flag;
    const char* opts[] = {arch_opt.c_str()};

    hiprtcResult comp = hiprtcCompileProgram(prog, 1, opts);
    if (comp != HIPRTC_SUCCESS) {
        size_t logSz; hiprtcGetProgramLogSize(prog, &logSz);
        char* log = new char[logSz]; hiprtcGetProgramLog(prog, log);
        printf("  %-35s  COMPILE FAIL: %s\n", label, log);
        delete[] log; hiprtcDestroyProgram(&prog);
        return false;
    }

    size_t codeSz;
    RTC_CHECK(hiprtcGetCodeSize(prog, &codeSz));
    char* code = new char[codeSz];
    RTC_CHECK(hiprtcGetCode(prog, code));
    RTC_CHECK(hiprtcDestroyProgram(&prog));

    hipModule_t mod;
    hipFunction_t func;
    hipError_t err = hipModuleLoadData(&mod, code);
    if (err != hipSuccess) {
        printf("  %-35s  LOAD FAIL: %s\n", label, hipGetErrorString(err));
        delete[] code;
        return false;
    }
    HIP_CHECK(hipModuleGetFunction(&func, mod, func_name));

    float *d_x, *d_y;
    float* h_x = new float[N];
    float* h_y = new float[N];
    for (int i = 0; i < N; i++) { h_x[i] = 1.0f; h_y[i] = 2.0f; }

    HIP_CHECK(hipMalloc(&d_x, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_x, h_x, N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y, N * sizeof(float), hipMemcpyHostToDevice));

    int n = N;
    void* args[] = {&d_y, &d_x, &alpha, &n};
    HIP_CHECK(hipModuleLaunchKernel(func, (N+255)/256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("  %-35s  EXEC FAIL: %s\n", label, hipGetErrorString(err));
        hipFree(d_x); hipFree(d_y); delete[] h_x; delete[] h_y;
        hipModuleUnload(mod); delete[] code;
        return false;
    }

    HIP_CHECK(hipMemcpy(h_y, d_y, N * sizeof(float), hipMemcpyDeviceToHost));

    double maxErr = 0;
    for (int i = 0; i < N; i++)
        maxErr = fmax(maxErr, fabs(h_y[i] - 5.0f));

    bool pass = maxErr < 1e-6;
    printf("  %-35s  maxErr=%e  %s\n", label, maxErr, pass ? "PASS" : "FAIL");

    hipModuleUnload(mod); hipFree(d_x); hipFree(d_y);
    delete[] h_x; delete[] h_y; delete[] code;
    return pass;
}

int main()
{
    const char* xnack = getenv("HSA_XNACK");
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("HSA_XNACK=%s  arch=%s\n\n", xnack ? xnack : "unset", prop.gcnArchName);

    const int N = 1 << 20;

    rtc_compile_and_run("RTC normal, gfx942",          kernel_src_normal, "saxpy_normal",  "gfx942", N, 3.0f);
    rtc_compile_and_run("RTC buffer_intrinsic, gfx942", kernel_src_buffer, "saxpy_buffer", "gfx942", N, 3.0f);
}
