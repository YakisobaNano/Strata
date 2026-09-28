// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>

namespace strata::prefill {
namespace {

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

bool needs_bf16_fp32_fallback(bool& fallback, std::string& err) {
    int device = 0;
    cudaDeviceProp prop{};
    cudaError_t e = cudaGetDevice(&device);
    if (e == cudaSuccess) e = cudaGetDeviceProperties(&prop, device);
    if (e != cudaSuccess) { err = std::string("prefill gemm: CUDA device query: ") + cudaGetErrorString(e); return false; }
    fallback = prop.major < 8 && std::getenv("STRATA_FORCE_BF16_FP32_GEMM") != nullptr;
    return true;
}

__global__ void bf16_to_fp32_kernel(const uint16_t* __restrict__ src, float* __restrict__ dst, int64_t n) {
    int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] = __uint_as_float((uint32_t) src[i] << 16);
}

void bf16_to_fp32(const uint16_t* src, float* dst, int64_t n, cudaStream_t stream) {
    constexpr int threads = 256;
    const int64_t needed = (n + threads - 1) / threads;
    const unsigned blocks = (unsigned) (needed < 65535 ? needed : 65535);
    bf16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(src, dst, n);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill gemm: BF16-to-FP32 conversion: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

Gemm::~Gemm() {
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    if (!needs_bf16_fp32_fallback(bf16_fp32_fallback_, err)) return false;
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { err = "prefill gemm: cublasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    cublasSetStream(h, (cudaStream_t) stream);
    workspace_ = workspace;
    cublasSetWorkspace(h, workspace_, ws_bytes);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    return true;
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    if (!needs_bf16_fp32_fallback(bf16_fp32_fallback_, err)) return false;
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { err = "prefill gemm: cublasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    cublasSetStream(h, (cudaStream_t) stream);
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (cudaMalloc(&workspace_, ws) != cudaSuccess) { err = "prefill gemm: workspace"; return false; }
    cublasSetWorkspace(h, workspace_, ws);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    if (scratch_elems > 0 && cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2) != cudaSuccess) {
        err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB";
        return false;
    }
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
    if (!bf16_fp32_fallback_) {
        // cuBLAS 12.8 accepts this BF16 input GEMM on V100 as well as newer devices. If a future device/library
        // combination rejects it, use the explicit FP32 fallback below.
        const cublasStatus_t status = cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            (int) N, (int) T, (int) K, &alpha, W, CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) K,
            &beta, Y, CUDA_R_32F, (int) ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (status == CUBLAS_STATUS_SUCCESS) return;
        if (status != CUBLAS_STATUS_NOT_SUPPORTED && status != CUBLAS_STATUS_ARCH_MISMATCH) ck(status, "cublasGemmEx");
    }

    {
        // Widen exact raw BF16 values and use SGEMM in bounded row/token tiles, keeping peak scratch fixed
        // instead of expanding a potentially very large weight matrix.
        const int64_t capacity = scratch_elems_ / 2;  // scratch is allocated as uint16_t; reinterpret half as FP32.
        if (!scratch_ || K <= 0 || K > std::numeric_limits<int>::max() ||
            capacity / K < 2 || T > std::numeric_limits<int>::max() || N > std::numeric_limits<int>::max()) {
            std::fprintf(stderr, "prefill gemm: Volta BF16 fallback scratch/shape is too small (T=%lld N=%lld K=%lld)\n",
                         (long long) T, (long long) N, (long long) K);
            std::exit(1);
        }
        const int64_t max_t = capacity / (2 * K);
        const int64_t tile_t = max_t < T ? max_t : T;
        float* converted_x = (float*) scratch_;
        for (int64_t t0 = 0; t0 < T; t0 += tile_t) {
            const int64_t nt = T - t0 < tile_t ? T - t0 : tile_t;
            const int64_t x_count = nt * K;
            bf16_to_fp32(X + t0 * K, converted_x, x_count, (cudaStream_t) stream_);
            const int64_t tile_n_capacity = (capacity - x_count) / K;
            const float* converted_w = converted_x + x_count;
            for (int64_t n0 = 0; n0 < N;) {
                const int64_t nn = N - n0 < tile_n_capacity ? N - n0 : tile_n_capacity;
                bf16_to_fp32(W + n0 * K, (float*) converted_w, nn * K, (cudaStream_t) stream_);
                ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) nn, (int) nt, (int) K,
                               &alpha, converted_w, (int) K, converted_x, (int) K, &beta,
                               Y + t0 * ldy + n0, (int) ldy), "cublasSgemm BF16 fallback");
                n0 += nn;
            }
        }
        return;
    }
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
