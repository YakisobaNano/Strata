#include "strata/kernels/bf16_bits.hpp"
#include "strata/prefill/gemm.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "bf16_gemm_parity: %s: %s\n", where, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T> T* upload(const std::vector<T>& src) {
    T* p = nullptr;
    ck(cudaMalloc(&p, src.size() * sizeof(T)), "cudaMalloc");
    ck(cudaMemcpy(p, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}

bool force_fp32_fallback(bool enabled) {
#if defined(_WIN32)
    return _putenv_s("STRATA_FORCE_BF16_FP32_GEMM", enabled ? "1" : "") == 0;
#else
    return enabled ? setenv("STRATA_FORCE_BF16_FP32_GEMM", "1", 1) == 0
                   : unsetenv("STRATA_FORCE_BF16_FP32_GEMM") == 0;
#endif
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) != "--selftest") {
        std::fprintf(stderr, "usage: bf16_gemm_parity [--selftest]\n");
        return 2;
    }
    constexpr int T = 3, N = 5, K = 7, LDY = 8;
    constexpr float beta = 0.25f;
    std::vector<uint16_t> x(T * K), w(N * K);
    for (int i = 0; i < T * K; ++i) x[i] = strata::kernels::bf16_from_f32((i - 9) * 0.137f);
    for (int i = 0; i < N * K; ++i) w[i] = strata::kernels::bf16_from_f32((i % 11 - 5) * 0.083f);
    std::vector<float> y(LDY * T);
    for (int i = 0; i < (int) y.size(); ++i) y[i] = -0.5f + i * 0.03125f;
    const std::vector<float> initial = y;

    uint16_t* d_x = upload(x);
    uint16_t* d_w = upload(w);
    float* d_y = upload(y);
    cudaStream_t stream{};
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
    std::string err;
    if (!force_fp32_fallback(true)) {
        std::fprintf(stderr, "bf16_gemm_parity: could not force the FP32 fallback for its parity check\n");
        return 1;
    }
    {
        // Small scratch deliberately forces both token and output-row tiling in the Volta fallback.
        strata::prefill::Gemm gemm;
        if (!gemm.init(stream, 64, err)) {
            std::fprintf(stderr, "bf16_gemm_parity: Gemm init: %s\n", err.c_str());
            return 1;
        }
        gemm.bf16(d_x, d_w, d_y, T, N, K, LDY, beta);
        ck(cudaStreamSynchronize(stream), "GEMM sync");
        ck(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost), "readback");
    }
    if (!force_fp32_fallback(false)) {
        std::fprintf(stderr, "bf16_gemm_parity: could not clear the FP32 fallback override\n");
        return 1;
    }

    double worst = 0.0;
    for (int t = 0; t < T; ++t) {
        for (int n = 0; n < N; ++n) {
            double dot = 0.0;
            for (int k = 0; k < K; ++k)
                dot += (double) strata::kernels::f32_from_bf16(x[t * K + k]) *
                       strata::kernels::f32_from_bf16(w[n * K + k]);
            const double expected = dot + beta * initial[t * LDY + n];
            const double error = std::abs((double) y[t * LDY + n] - expected);
            worst = std::max(worst, error / std::max(1.0, std::abs(expected)));
            if (error > 2e-5 * std::max(1.0, std::abs(expected))) {
                std::fprintf(stderr, "bf16_gemm_parity: t=%d n=%d got %.9g expected %.9g\n",
                             t, n, y[t * LDY + n], expected);
                return 1;
            }
        }
        for (int n = N; n < LDY; ++n) {
            if (y[t * LDY + n] != initial[t * LDY + n]) {
                std::fprintf(stderr, "bf16_gemm_parity: output padding was modified\n");
                return 1;
            }
        }
    }
    cudaDeviceProp prop{};
    int device = 0;
    ck(cudaGetDevice(&device), "device query");
    ck(cudaGetDeviceProperties(&prop, device), "device properties");

    // Probe the exact CUDA_R_16BF/CUBLAS_COMPUTE_32F API used before the Volta dispatch at a representative
    // aligned dense-projection size. This distinguishes architecture support from tiny-matrix selection.
    constexpr int PT = 4, PN = 2560, PK = 10240;
    std::vector<uint16_t> px(PT * PK, strata::kernels::bf16_from_f32(0.25f));
    std::vector<uint16_t> pw(PN * PK, strata::kernels::bf16_from_f32(0.25f));
    std::vector<float> py(PN * PT, 0.5f), py_out(PN * PT);
    uint16_t* d_px = upload(px);
    uint16_t* d_pw = upload(pw);
    float* d_py = upload(py);
    cublasHandle_t probe{};
    cublasStatus_t probe_status = cublasCreate(&probe);
    if (probe_status == CUBLAS_STATUS_SUCCESS) {
        probe_status = cublasSetStream(probe, stream);
        if (probe_status == CUBLAS_STATUS_SUCCESS) {
            const float alpha = 1.0f, probe_beta = 0.25f;
            probe_status = cublasGemmEx(probe, CUBLAS_OP_T, CUBLAS_OP_N, PN, PT, PK, &alpha,
                                        d_pw, CUDA_R_16BF, PK, d_px, CUDA_R_16BF, PK, &probe_beta,
                                        d_py, CUDA_R_32F, PN, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
    }
    if (probe_status == CUBLAS_STATUS_SUCCESS) {
        const cudaError_t sync = cudaStreamSynchronize(stream);
        if (sync == cudaSuccess) {
            ck(cudaMemcpy(py_out.data(), d_py, py_out.size() * sizeof(float), cudaMemcpyDeviceToHost), "BF16 API readback");
            for (float value : py_out) {
                if (std::abs(value - 640.125f) > 1e-2f) {
                    std::fprintf(stderr, "bf16_gemm_parity: direct CUDA_R_16BF cuBLAS result %.9g, expected 640.125\n", value);
                    return 1;
                }
            }
        } else {
            if (prop.major >= 8) ck(sync, "direct CUDA_R_16BF cuBLAS sync");
            probe_status = CUBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    if (probe_status != CUBLAS_STATUS_SUCCESS && prop.major >= 8) {
        std::fprintf(stderr, "bf16_gemm_parity: direct CUDA_R_16BF cuBLAS status %d on sm_%d%d\n",
                     (int) probe_status, prop.major, prop.minor);
        return 1;
    }
    ck(cudaMemcpy(d_py, py.data(), py.size() * sizeof(float), cudaMemcpyHostToDevice), "reset projection output");
    {
        // Exercise the production Gemm dispatch with the same projected shape and full-size production scratch.
        strata::prefill::Gemm gemm;
        if (!gemm.init(stream, 32ll << 20, err)) {
            std::fprintf(stderr, "bf16_gemm_parity: production Gemm init: %s\n", err.c_str());
            return 1;
        }
        gemm.bf16(d_px, d_pw, d_py, PT, PN, PK, PN, beta);
        ck(cudaStreamSynchronize(stream), "production BF16 GEMM sync");
        ck(cudaMemcpy(py_out.data(), d_py, py_out.size() * sizeof(float), cudaMemcpyDeviceToHost),
           "production BF16 GEMM readback");
    }
    for (float value : py_out) {
        if (std::abs(value - 640.125f) > 1e-2f) {
            std::fprintf(stderr, "bf16_gemm_parity: production Gemm result %.9g, expected 640.125\n", value);
            return 1;
        }
    }
    if (probe) cublasDestroy(probe);
    cudaFree(d_py); cudaFree(d_pw); cudaFree(d_px);
    cudaFree(d_y); cudaFree(d_w); cudaFree(d_x);
    ck(cudaStreamDestroy(stream), "stream destroy");
    std::printf("bf16_gemm_parity: OK on %s (sm_%d%d), forced FP32 tiling PASS; production BF16 dispatch %s, direct CUDA_R_16BF cuBLAS status %d, max relative error %.3g\n",
                prop.name, prop.major, prop.minor, probe_status == CUBLAS_STATUS_SUCCESS ? "cuBLAS" : "FP32 fallback", (int) probe_status, worst);
    return 0;
}
