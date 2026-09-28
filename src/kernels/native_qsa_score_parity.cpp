#include "strata/kernels/native_qsa_score.hpp"

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
        std::fprintf(stderr, "native_qsa_score_parity: %s: %s\n", where, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T> T* device_copy(const std::vector<T>& src, cudaStream_t stream) {
    T* p = nullptr;
    ck(cudaMalloc(&p, src.size() * sizeof(T)), "cudaMalloc");
    ck(cudaMemcpyAsync(p, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice, stream), "upload");
    return p;
}

float reference_score(const float* pooled, const float* query, const float* bias, int row, int n) {
    float h[4]{};
    for (int head = 0; head < 4; ++head) {
        double sum = 0.0;
        for (int k = 0; k < 128; ++k)
            sum += (double) pooled[(size_t) row * 128 + k] * query[head * 128 + k];
        h[head] = std::fmax((float) sum, 0.0f);
    }
    float score = ((h[0] + h[1]) + h[2]) + h[3];
    score += bias[row];
    if (row == n / 4 && n % 4) score += 1e9f;
    return score + 0.0f;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) != "--selftest") {
        std::fprintf(stderr, "usage: native_qsa_score_parity [--selftest]\n");
        return 2;
    }
    constexpr int max_cells = 17, max_blocks = max_cells / 4 + 1;
    const auto shapes = strata::kernels::qsa_real_shapes();
    std::vector<float> pooled((size_t) max_blocks * 128), query(4 * 128), bias(max_blocks), output(max_cells, -1234.0f);
    for (int row = 0; row < max_blocks; ++row) {
        bias[row] = (row - 2) * 0.125f;
        for (int k = 0; k < 128; ++k)
            pooled[(size_t) row * 128 + k] = std::sin(0.013f * (row + 1) * (k + 3));
    }
    for (int h = 0; h < 4; ++h)
        for (int k = 0; k < 128; ++k) query[h * 128 + k] = std::cos(0.017f * (h + 2) * (k + 1));

    cudaStream_t stream{};
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
    float* d_pooled = device_copy(pooled, stream);
    float* d_query = device_copy(query, stream);
    float* d_bias = device_copy(bias, stream);
    float* d_output = nullptr;
    int32_t* d_step = nullptr;
    ck(cudaMalloc(&d_output, output.size() * sizeof(float)), "output allocation");
    ck(cudaMalloc(&d_step, strata::kernels::kStepCount * sizeof(int32_t)), "step allocation");

    double worst = 0.0;
    for (const int n : {16, 17}) {
        int32_t step[strata::kernels::kStepCount]{};
        strata::kernels::qsa_step_fill(step, n - 1, shapes);
        ck(cudaMemcpyAsync(d_step, step, sizeof(step), cudaMemcpyHostToDevice, stream), "step upload");
        ck(cudaMemcpyAsync(d_output, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
           "output init");
        strata::kernels::native_qsa_score(d_pooled, d_query, d_bias, shapes, d_step, max_blocks, max_cells,
                                          d_output, stream);
        ck(cudaMemcpyAsync(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
           "output readback");
        ck(cudaStreamSynchronize(stream), "stream sync");
        for (int cell = 0; cell < n; ++cell) {
            const int row = cell / 4;
            const float expected = reference_score(pooled.data(), query.data(), bias.data(), row, n);
            const double error = std::abs((double) output[cell] - expected);
            worst = std::max(worst, error / std::max(1.0, std::abs((double) expected)));
            if (error > 2e-3 * std::max(1.0, std::abs((double) expected))) {
                std::fprintf(stderr, "native_qsa_score_parity: n=%d cell=%d got %.9g expected %.9g\n",
                             n, cell, output[cell], expected);
                return 1;
            }
        }
    }
    cudaFree(d_step); cudaFree(d_output); cudaFree(d_bias); cudaFree(d_query); cudaFree(d_pooled);
    ck(cudaStreamDestroy(stream), "stream destroy");
    cudaDeviceProp prop{};
    int device = 0;
    ck(cudaGetDevice(&device), "device query");
    ck(cudaGetDeviceProperties(&prop, device), "device properties");
    std::printf("native_qsa_score_parity: OK on %s (sm_%d%d), max relative error %.3g\n",
                prop.name, prop.major, prop.minor, worst);
    return 0;
}
