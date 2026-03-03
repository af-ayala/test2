/*
hipcc -O3 --offload-arch=gfx942 -o xnack_rtc_test xnack_rtc_test.cpp -lhiprtc

HSA_XNACK=0 ./xnack_rtc_test
HSA_XNACK=1 ./xnack_rtc_test
*/

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <cstdio>
#include <cmath>
#include <cstring>

#define HIP_CHECK(x) do { hipError_t e = (x); if(e) { fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1); } } while(0)
#define RTC_CHECK(x) do { hiprtcResult r = (x); if(r) { fprintf(stderr,"HIPRTC %s:%d %s\n",__FILE__,__LINE__,hiprtcGetErrorString(r)); exit(1); } } while(0)

const char* kernel_src = R"(
extern "C" __global__ void saxpy(float* y, const float* x, float a, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = a * x[i] + y[i];
}
)";

int main()
{
    const char* xnack = getenv("HSA_XNACK");
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("HSA_XNACK=%s  arch=%s\n", xnack ? xnack : "unset", prop.gcnArchName);

    // --- RTC compile for bare gfx942 (stripped, like rocFFT does) ---
    hiprtcProgram prog;
    RTC_CHECK(hiprtcCreateProgram(&prog, kernel_src, "saxpy.cu", 0, nullptr, nullptr));

    const char* opts[] = {"--offload-arch=gfx942"};
    printf("RTC compiling with: %s\n", opts[0]);

    hiprtcResult compResult = hiprtcCompileProgram(prog, 1, opts);
    if (compResult != HIPRTC_SUCCESS) {
        size_t logSize;
        hiprtcGetProgramLogSize(prog, &logSize);
        char* log = new char[logSize];
        hiprtcGetProgramLog(prog, log);
        fprintf(stderr, "Compile failed:\n%s\n", log);
        delete[] log;
        return 1;
    }

    size_t codeSize;
    RTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
    char* code = new char[codeSize];
    RTC_CHECK(hiprtcGetCode(prog, code));
    RTC_CHECK(hiprtcDestroyProgram(&prog));

    hipModule_t module;
    hipFunction_t func;
    HIP_CHECK(hipModuleLoadData(&module, code));
    HIP_CHECK(hipModuleGetFunction(&func, module, "saxpy"));

    // --- Run kernel ---
    const int N = 1 << 20;
    float *d_x, *d_y;
    float* h_x = new float[N];
    float* h_y = new float[N];
    for (int i = 0; i < N; i++) { h_x[i] = 1.0f; h_y[i] = 2.0f; }

    HIP_CHECK(hipMalloc(&d_x, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_x, h_x, N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y, N * sizeof(float), hipMemcpyHostToDevice));

    float alpha = 3.0f;
    int n = N;
    void* args[] = {&d_y, &d_x, &alpha, &n};
    HIP_CHECK(hipModuleLaunchKernel(func, (N+255)/256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_y, d_y, N * sizeof(float), hipMemcpyDeviceToHost));

    double maxErr = 0;
    for (int i = 0; i < N; i++)
        maxErr = fmax(maxErr, fabs(h_y[i] - 5.0f));

    printf("RTC bare gfx942:  maxErr=%e  %s\n", maxErr, maxErr < 1e-6 ? "PASS" : "FAIL");

    // --- Now try RTC with explicit xnack+ ---
    RTC_CHECK(hiprtcCreateProgram(&prog, kernel_src, "saxpy.cu", 0, nullptr, nullptr));
    const char* opts2[] = {"--offload-arch=gfx942:xnack+"};
    printf("RTC compiling with: %s\n", opts2[0]);

    compResult = hiprtcCompileProgram(prog, 1, opts2);
    if (compResult != HIPRTC_SUCCESS) {
        size_t logSize;
        hiprtcGetProgramLogSize(prog, &logSize);
        char* log = new char[logSize];
        hiprtcGetProgramLog(prog, log);
        fprintf(stderr, "Compile failed:\n%s\n", log);
        delete[] log;
        return 1;
    }

    RTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
    delete[] code;
    code = new char[codeSize];
    RTC_CHECK(hiprtcGetCode(prog, code));
    RTC_CHECK(hiprtcDestroyProgram(&prog));

    HIP_CHECK(hipModuleUnload(module));
    HIP_CHECK(hipModuleLoadData(&module, code));
    HIP_CHECK(hipModuleGetFunction(&func, module, "saxpy"));

    HIP_CHECK(hipMemcpy(d_x, h_x, N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y, N * sizeof(float), hipMemcpyHostToDevice));
    for (int i = 0; i < N; i++) h_y[i] = 2.0f;
    HIP_CHECK(hipMemcpy(d_y, h_y, N * sizeof(float), hipMemcpyHostToDevice));

    HIP_CHECK(hipModuleLaunchKernel(func, (N+255)/256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_y, d_y, N * sizeof(float), hipMemcpyDeviceToHost));

    maxErr = 0;
    for (int i = 0; i < N; i++)
        maxErr = fmax(maxErr, fabs(h_y[i] - 5.0f));

    printf("RTC gfx942:xnack+: maxErr=%e  %s\n", maxErr, maxErr < 1e-6 ? "PASS" : "FAIL");

    HIP_CHECK(hipModuleUnload(module));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    delete[] h_x; delete[] h_y; delete[] code;
}
